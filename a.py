from prepare import *


class modelKD:
    def __init__(self, args, device):
        self.device = device
        self.p = args.p
        self.lamda = args.lamda
        self.student = self.loadModel(args.network)
        self.optimizer = optim.SGD(self.student.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def loadModel(self, name):
        if name == "resnet18":
            r = resnet18(num_classes=10)
        elif name == "resnet34":
            r = resnet34(num_classes=10)
        elif name == "vgg16":
            r = vgg16(num_classes=10)
        elif name == "mobilenet_v2":
            r = mobilenet_v2(num_classes=10)
        else:
            raise ValueError("network name error")
        return r.to(self.device)

    def addNoise(self, num_models=4):
        model = self.student
        noisy_models = []

        for _ in range(num_models):
            noisy_model = copy.deepcopy(model)
        for layer in noisy_model.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                for param in layer.parameters():
                    # 添加均值为0，方差为all_params_std * lamda的正态分布噪声
                    all_params = param.view(-1)
                    all_params_std = torch.std(all_params)
                    noise = torch.randn_like(param) * all_params_std * math.sqrt(self.lamda)
                    param.data.add_(noise)
                    mask = torch.bernoulli(torch.full_like(param, self.p))
                    # 将参数设置为0，以实现dropout的效果
                    param.data.mul_(mask)
        noisy_model = noisy_model.to(self.device)
        noisy_model.eval()
        noisy_models.append(noisy_model)
        self.teachers = noisy_models


def loadData(args):
    clean_train_data = torchvision.datasets.CIFAR10(root=args.datadir, train=True, download=True)
    clean_train_data = MakePoison(clean_train_data, args.attack, args.pr_A, args.target, "train")
    clean_train_dataloader = DataLoader(clean_train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    clean_test_data = torchvision.datasets.CIFAR10(
        root=args.datadir,
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ]
        ),
    )
    clean_test_dataloader = DataLoader(clean_test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    poison_train_data = PoisonDataset(
        clean_train_data,
        np.random.choice(len(clean_train_data), int(args.pr_G * len(clean_train_data)), replace=False),
        target=args.target,
    )
    poison_train_dataloader = DataLoader(poison_train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)

    poison_test_data = PoisonDataset(
        clean_test_data,
        np.random.choice(len(clean_test_data), len(clean_test_data), replace=False),
        target=args.target,
    )
    poison_test_dataloader = DataLoader(poison_test_data, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    return clean_train_dataloader, clean_test_dataloader, poison_train_dataloader, poison_test_dataloader


def tGenerate(models, tri, trigger_optimizer, device):
    masks = []
    triggers = []
    tri.train()
    (models.student).eval()
    for x, y in tqdm(poison_train_dataloader):
        x = x.to(device)
        y = y.to(device)
        x = tri(x)
        logits = (models.student)(x)
        loss = student_lambda_t * F.cross_entropy(logits, y) + student_lambda_mask * torch.norm(tri.mask, p=2)
        trigger_optimizer.zero_grad()
        loss.backward()
        trigger_optimizer.step()
        with torch.no_grad():
            tri.mask.clamp_(0, 1)
            tri.trigger.clamp_(-1, 1)
    masks.append(tri.mask.clone())
    triggers.append(tri.trigger.clone())

    global teacher_lambda_mask
    models.addNoise()
    squared_sum = []
    for teacher in models.teachers:
        for x, y in tqdm(poison_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            x = tri(x)
            logits = teacher(x)
            loss = teacher_lambda_t * F.cross_entropy(logits, y) + teacher_lambda_mask * torch.norm(tri.mask, p=2)
            trigger_optimizer.zero_grad()
            loss.backward()

            squared_sum.append(torch.sum(tri.trigger.grad**2).item())
            trigger_optimizer.step()
            with torch.no_grad():
                tri.mask.clamp_(0, 1)
                tri.trigger.clamp_(-1, 1)
        masks.append(tri.mask.clone())
        triggers.append(tri.trigger.clone())
        avg = sum(squared_sum) / len(squared_sum)
        # print(avg)
        if avg > 2e-7:
            # 如果 F.cross_entropy(logits, y) 梯度较大，减小 student_lambda_mask
            teacher_lambda_mask = teacher_lambda_mask * 0.9
        else:
            # 如果 F.cross_entropy(logits, y) 梯度较小，增大 student_lambda_mask
            teacher_lambda_mask = teacher_lambda_mask * 1.1

    average_mask = torch.mean(torch.stack(masks), dim=0)
    average_trigger = torch.mean(torch.stack(triggers), dim=0)
    tri.mask.data = args.beita * tri.mask.data + (1 - args.beita) * average_mask
    tri.trigger.data = args.beita * tri.trigger.data + (1 - args.beita) * average_trigger


def trainAgenerate(args, models, tri, trigger_optimizer, device):
    epochs = args.epochs
    for epoch in range(epochs):
        print("epoch: {}".format(epoch))

        (models.student).train()

        for x, y in tqdm(clean_train_dataloader):
            x = x.to(device)
            y = y.to(device)
            student_logits = (models.student)(x)
            hard_loss = F.cross_entropy(student_logits, y)
            # if epoch > 10:
            #     models.addNoise()
            #     soft_loss = 0
            #     for teacher in models.teachers:
            #         with torch.no_grad():
            #             teacher_logits = (teacher)(x)
            #             soft_loss += F.kl_div(
            #                 F.log_softmax(student_logits / args.temperature, dim=1),
            #                 F.softmax(teacher_logits / args.temperature, dim=1),
            #                 reduction="batchmean",
            #             )
            #     loss = args.alpha * soft_loss + (1 - args.alpha) * hard_loss
            #     # 可以加个epoch/epochs的参数，随着训练周期的增加，学生模型应更贴近教师模型
            # else:
            #     loss = hard_loss
            models.addNoise()
            soft_loss = 0
            for teacher in models.teachers:
                with torch.no_grad():
                    teacher_logits = (teacher)(x)
                    soft_loss += F.kl_div(
                        F.log_softmax(student_logits / args.temperature, dim=1),
                        F.softmax(teacher_logits / args.temperature, dim=1),
                        reduction="batchmean",
                    )
            alpha = args.alpha * epoch / epochs
            loss = alpha * soft_loss + (1 - alpha) * hard_loss

            models.optimizer.zero_grad()
            loss.backward()
            models.optimizer.step()

        models.scheduler.step()

        tGenerate(models, tri, trigger_optimizer, device)

        (models.student).eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(clean_test_dataloader):
                x = x.to(device)
                y = y.to(device)
                logits = (models.student)(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
        accuracy = correct / total
        print("student ACC:", accuracy)

        tri.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in tqdm(poison_test_dataloader):
                x = x.to(device)
                x = tri(x)
                y = y.to(device)
                logits = (models.student)(x)
                _, predict_label = logits.max(1)
                total += y.size(0)
                correct += predict_label.eq(y).sum().item()
        asr = correct / total
        print("student ASR:", asr)

        wandb.log({"acc": accuracy, "asr": asr})

    model_folder_path = "new_model/" + args.attack
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    torch.save(models.student.state_dict(), os.path.join(model_folder_path, "{}.pth".format(args.network)))

    tri_folder_path = "new_trigger/" + args.attack
    if not os.path.exists(tri_folder_path):
        os.makedirs(tri_folder_path)
    torch.save(tri.state_dict(), os.path.join(tri_folder_path, "tri_from_{}.pth".format(args.network)))


def main():
    models = modelKD(args, DEVICE)
    tri = Trigger(size=32).to(DEVICE)
    trigger_optimizer = optim.Adam(tri.parameters(), lr=1e-2)
    trainAgenerate(args, models, tri, trigger_optimizer, DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("--datadir", default="./dataset", help="root directory of data")
    parser.add_argument("--gpu", default="0", help="gpu id")
    parser.add_argument("--dataset", default="cifar10", help="dataset")
    parser.add_argument("--network", default="resnet34", help="network structure")
    parser.add_argument("--attack", default="None", help="attack way")
    parser.add_argument("--batch_size", type=int, default=128, help="attack size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--target", type=int, default=1, help="target label")
    parser.add_argument("--pr_G", type=float, default=0.1, help="poisoning rate for generate")
    parser.add_argument("--pr_A", type=float, default=0.1, help="poisoning rate for attack")
    parser.add_argument("--lamda", type=float, default=0.2, help="正态分布前的参数")
    parser.add_argument("--p", type=float, default=0.85, help="伯努利分布")
    parser.add_argument("--temperature", type=float, default=1.5, help="KD温度系数")
    parser.add_argument("--alpha", type=float, default=0.2, help="soft loss and hard loss")
    parser.add_argument("--beita", type=float, default=0.3, help="trigger迭代时,旧值占比")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DEVICE = torch.device("cuda")

    clean_train_dataloader, clean_test_dataloader, poison_train_dataloader, poison_test_dataloader = loadData(args)

    student_lambda_t = 1e-1
    student_lambda_mask = 1e-3

    teacher_lambda_t = 1e-2
    teacher_lambda_mask = 1e-2

    wandb.init(
        # set the wandb project where this run will be logged
        project="for generate",
        # track hyperparameters and run metadata
        config={
            "architecture": args.network,
            "attack": args.attack,
        },
    )

    time_start = time.time()
    main()
    time_end = time.time()
    print("=" * 50)
    print("Running time:", (time_end - time_start) / 60, "m")
    print("=" * 50)
