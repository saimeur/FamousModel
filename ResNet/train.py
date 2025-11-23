import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import ResNet50
from torchvision.models import resnet50

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--torch_resnet', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--epoch', type=int, default=3)
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),
    ])

    # Charger le dataset d'entraînement et de test
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Hyperparameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-3
    EPOCH = args.epoch
    TRAIN_MODEL = args.train
    TEST_MODEL = args.test
    SAVE_MODEL = args.save
    LOAD_MODEL = args.load
    SUFFIXE = "torch" if args.torch_resnet else "perso"

    data, targets = next(iter(trainloader))
    print(data.shape, targets.shape)
    # imgs_grid = torchvision.utils.make_grid(data, normalize=True, scale_each=True)
    # plt.imshow(imgs_grid.permute(1, 2, 0))
    # plt.show()

    # Choose model
    if args.torch_resnet:
        model = resnet50(weights=None)  # ⚠️ pas pré-entraîné
        model.fc = nn.Linear(model.fc.in_features, 10)  # adapter la dernière couche
    else:
        model = ResNet50(10)
    model.to(device=DEVICE)
    
    if TRAIN_MODEL:
        model.train()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

        loss_over_epochs = [] 
        for epoch in range(EPOCH):
            loop = tqdm(trainloader)
            tot_loss = 0
            for batch_idx, (data, targets) in enumerate(loop):
            # loop = tqdm(range(50))
            # for i in loop:
                data = data.to(device=DEVICE)
                targets = targets.to(device=DEVICE)

                optimizer.zero_grad()
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                tot_loss += loss.item()
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())

            tot_loss /= float(batch_idx + 1)
            loss_over_epochs.append(tot_loss)
            print(f"Avg loss over epoch {epoch} : {tot_loss}")

        plt.figure(figsize=(8, 5))
        plt.plot(loss_over_epochs, marker='o')
        plt.title('Loss over each epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)  # Ajoute une grille pour la lecture
        plt.tight_layout()
        plt.savefig(SUFFIXE + "_loss_curve.png")
        plt.show()


    if SAVE_MODEL:
        checkpoints = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        print("=> Saving checkpoint")
        filename = SUFFIXE + "_resnet50.pth.tar"
        torch.save(checkpoints, filename)

    if LOAD_MODEL:
        print("=> Loading checkpoint")
        filename = SUFFIXE + "_resnet50.pth.tar"
        checkpoints = torch.load(filename)
        model.load_state_dict(checkpoints["state_dict"])


    if TEST_MODEL:
        model.eval()
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for _ , (data, targets) in enumerate(tqdm(testloader)):
                data = data.to(device=DEVICE)
                targets = targets.to(device=DEVICE)
                _, preds = torch.max(model(data), dim=1)
                num_correct += (preds==targets).sum().item()
                num_samples += preds.shape[0]
        print(
            f"Got {num_correct} / {num_samples}"
            f"With accuracy of : {float(num_correct)/float(num_samples)}")
        