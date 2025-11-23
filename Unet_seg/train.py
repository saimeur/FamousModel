import torch
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import UNET
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_transform,
    )

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 2
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = False
TRAIN_MODEL = False
TENSORBOARD = True
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler,
             epoch=None, lr=None, batch_size=None, writer=None):
    loop = tqdm(loader)
    losses = []
    accuracies = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        # autocast to use float16 and float32 -> faster training
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data)
            loss = loss_fn(predictions, targets)    
        losses.append(loss)
        
        # backward
        optimizer.zero_grad()
        ## scaler multiplies the loss by big factor to avoid vanishing gradient
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # compute accuracy
        output = (torch.sigmoid(predictions) > 0.5).float()
        num_correct = (output == targets).sum()
        tot_number = torch.numel(targets)
        accuracy = num_correct/tot_number
        accuracies.append(accuracy)

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # plot to tensorboard
        if writer is not None:
            step = batch_idx + epoch*len(loader)
            img_grid = torchvision.utils.make_grid(data, normalize=True, scale_each=True)
            targets_grid = torchvision.utils.make_grid(targets, normalize=True, scale_each=True)
            outputs_grid = torchvision.utils.make_grid(output, normalize=True, scale_each=True)
            writer.add_image('image', img_grid, step)
            writer.add_image('targets', targets_grid, step)
            writer.add_image('preds', outputs_grid, step)
            writer.add_histogram('last layer', model.final_conv.weight, step)
            writer.add_scalar('Training loss', loss, global_step= batch_idx + epoch*len(loader))
            writer.add_scalar('Training acc', accuracy, batch_idx + epoch*len(loader))
            # writer.add_embedding()  # plot PCA / t-SNE ...
    
    # writer.add_hparams({'lr': lr, 'bsize': batch_size},
    #                    {'accuracy': sum(accuracies)/len(accuracies),
    #                    'loss': sum(losses)/len(losses)},
    #                    )

def main():
    if TENSORBOARD :
        batch_sizes = [16]
        learning_rates = [0.001]
        heights = [160]
        widths  = [240]
        assert len(heights) == len(widths), 'Lenghts and widths need to have the same dimensions'
        for img_dim_idx in range(len(heights)):
            height, width = heights[img_dim_idx], widths[img_dim_idx]
            train_transform, val_transform = get_transform(img_height=height, img_width=width)
            for batch_size in batch_sizes:
                for lr in learning_rates:
                    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
                    loss_fn = nn.BCEWithLogitsLoss()    # assune target that has values in (0.0, 1.0)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    writer = SummaryWriter(log_dir= f"runs/UNET/plot data") if TENSORBOARD else None
                    # writer = SummaryWriter(log_dir= f"runs/UNET/Width {width} Height {height} BatchSize {batch_size} LR {lr}") if TENSORBOARD else None

                    train_loader, val_loader = get_loaders(
                        TRAIN_IMG_DIR,
                        TRAIN_MASK_DIR,
                        VAL_IMG_DIR,
                        VAL_MASK_DIR,
                        batch_size,
                        train_transform,
                        val_transform,
                        NUM_WORKERS,
                        PIN_MEMORY
                    )
                    if LOAD_MODEL:
                        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
                        check_accuracy(val_loader, model=model, device=DEVICE)

                    scaler = torch.amp.GradScaler(device=DEVICE)
                    for epoch in range(NUM_EPOCHS):
                        train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch, lr, batch_size,writer)
                
    else :
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_transform, val_transform = get_transform(IMAGE_HEIGHT, IMAGE_HEIGHT)
        train_loader, val_loader = get_loaders(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            VAL_IMG_DIR,
            VAL_MASK_DIR,
            BATCH_SIZE,
            train_transform,
            val_transform,
            NUM_WORKERS,
            PIN_MEMORY
        )
        if LOAD_MODEL:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
            check_accuracy(val_loader, model=model, device=DEVICE)

        scaler = torch.amp.GradScaler(device=DEVICE)
        for epoch in range(NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
            # save model
            checkpoints = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if SAVE_MODEL:
                save_checkpoint(checkpoints)

            # check accuracy
            check_accuracy(val_loader, model=model, device=DEVICE)

            # print some examples to a folder
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE
            )

if __name__ == "__main__":
    main()