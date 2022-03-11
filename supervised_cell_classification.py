import torch
from torch import nn
import classification_ModelNet40.models as models
import torch.backends.cudnn as cudnn
from classification_ModelNet40.models import pointMLP

from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from foldingnet import ReconstructionNet, ChamferLoss
from dataset import PointCloudDatasetAllBoth
import argparse
import os


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pointmlp-foldingnet")
    parser.add_argument("--dataset_path", default="/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/", type=str)
    parser.add_argument("--dataframe_path", default="/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv", type=str)
    parser.add_argument("--output_path", default="./", type=str)
    parser.add_argument("--num_epochs", default=250, type=int)
    parser.add_argument("--pmlp_ckpt_path", default="best_checkpoint.pth", type=str)

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    pmlp_ckpt_path = args.pmlp_ckpt_path

    name_net = output_path + "pointmlp_classifier"
    print("==> Building classifier...")
    net = pointMLP()
    device = "cuda"
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint_path = pmlp_ckpt_path
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["net"])
    for param in net.module.parameters():
        param.requires_grad = False
        
    new_embedding = nn.Linear(in_features=256, out_features=1, bias=True)
    net.module.classifier[8] = new_embedding
    net.module.classifier[8].weight.requires_grad = True
    net.module.classifier[8].bias.requires_grad = True
    net = net.to(device)
    print(net.module.classifier)

    batch_size = 16
    learning_rate = 0.00001
    dataset = PointCloudDatasetAllBoth(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        centring_only=True,
        cell_component="cell",
    )
    train_size = int(np.round(len(dataset) * 0.8))
    test_size = int(len(dataset) - train_size)

    train, valid = random_split(dataset, [train_size, test_size])

    dataloader_train = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(valid, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.module.parameters()),
        lr=learning_rate * 16 / batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    rec_loss = 0.0
    clus_loss = 0.0
    num_epochs = num_epochs
    net.train()
    threshold = 0.0
    losses = []
    test_acc = []
    best_acc = 0.0
    best_loss = 1000000000
    niter = 1
    for epoch in range(num_epochs):
        batch_num = 1
        train_loss = 0.0
        print("Training epoch {}".format(epoch))
        net.train()
        batches = []
        for i, data in enumerate(dataloader_train, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).type(torch.cuda.FloatTensor).to(device)
            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output = net(inputs.permute(0, 2, 1))
                optimizer.zero_grad()
                loss = criterion(output.type(torch.cuda.FloatTensor), 
                                 labels)
                acc = (torch.sigmoid(output).reshape(-1).detach().cpu().numpy().round() == labels.reshape(
                    -1
                ).detach().cpu().numpy().round()).mean()
                # ===================backward===================
                loss.backward()
                optimizer.step()

            train_loss += loss.detach().item() / batch_size
            batch_num += 1
            niter += 1

            lr = np.asarray(optimizer.param_groups[0]["lr"])
            if i % 10 == 0:
                print(
                    "[%d/%d][%d/%d]\tTrain loss: %.4f\t Train Acc: %.4f"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader_train),
                        loss.detach().item() / batch_size,
                        acc,
                    )
                )

        net.eval()
        valid_loss = 0.0
        for i, data in enumerate(dataloader_valid, 0):
            inputs, labels, _ = data
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).type(
                torch.cuda.FloatTensor
            ).to(device)
            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output = net(inputs.permute(0, 2, 1))
                loss = criterion(output.type(torch.cuda.FloatTensor),
                                 labels)
                acc = (torch.sigmoid(output).reshape(-1).detach().cpu().numpy().round() == labels.reshape(
                    -1).detach().cpu().numpy().round()).mean()

            valid_loss += loss.detach().item()
            
        print(
            f'Epoch {epoch} \t'
            f'\t Training Loss: {train_loss / len(dataloader_train)} \t'
            f'\t Validation Loss: {valid_loss / len(dataloader_valid)}')
        if valid_loss < best_loss:
            checkpoint = {
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": valid_loss,
            }
            best_loss = valid_loss
            create_dir_if_not_exist(output_path)
            print(
                "Saving model to:"
                + name_net
                + ".pt"
                + " with loss = {}".format(valid_loss)
                + " at epoch {}".format(epoch)
            )
            torch.save(checkpoint, name_net + ".pt")


