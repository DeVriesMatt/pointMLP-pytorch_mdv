import torch
from torch import nn
import classification_ModelNet40.models as models
import torch.backends.cudnn as cudnn
from classification_ScanObjectNN.models import pointMLPElite
from datetime import datetime
# from cell_dataset import PointCloudDatasetAllBoth
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from foldingnet import ReconstructionNet, ChamferLoss
from angle_loss import AngleLoss
from dataset import SimCLR1024Both
import logging

import argparse
import os
from tearing.folding_decoder import FoldingNetBasicDecoder
from nt_xent import NT_Xent
from reporting import get_experiment_name


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLR(nn.Module):
    def __init__(self, encoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.encoder.classifier[8] = Identity()
        self.projector = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 64, bias=False),
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pointmlp-simclr")
    parser.add_argument(
        "--dataset_path",
        default="/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/",
        type=str,
    )
    parser.add_argument(
        "--dataframe_path",
        default="/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv",
        type=str,
    )
    parser.add_argument("--output_path", default="./", type=str)
    parser.add_argument("--num_epochs", default=250, type=int)
    parser.add_argument(
        "--pmlp_ckpt_path", default="best_checkpoint_elite.pth", type=str
    )
    parser.add_argument(
        "--fold_ckpt_path",
        default="/home/mvries/Documents/GitHub/FoldingNetNew/nets/FoldingNetNew_50feats_planeshape_foldingdecoder_trainallTrue_centringonlyTrue_train_bothTrue_003.pt",
        type=str,
    )
    parser.add_argument(
        "--full_checkpoint_path",
        default="/home/mvries/Documents/GitHub/pointMLP-pytorch/"
                "best_checkpoint_elite.pth",
        type=str,
    )

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    pmlp_ckpt_path = args.pmlp_ckpt_path
    fold_ckpt_path = args.fold_ckpt_path
    full_checkpoint_path = args.full_checkpoint_path
    output_dir = output_path

    model_name = output_path + "pointmlpelite_simclr"
    print("==> Building encoder...")
    net = pointMLPElite(num_classes=15)
    device = "cuda"
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint_path = pmlp_ckpt_path
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["net"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("==> Building simclr...")

    model = SimCLR(encoder=net.module).cuda()

    data = torch.rand(2, 3, 1024).cuda()
    print("===> testing pointMLP simclr ...")
    # h_i, h_j, z_i, z_j = model(data)
    # print(h_i.shape)
    # print(z_i.shape)

    batch_size = 57
    learning_rate = 0.00003 * batch_size / 100
    dataset = SimCLR1024Both(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        centring_only=True,
        cell_component="cell",
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-8,
    )
    criterion = NT_Xent(batch_size, 0.5, 1)
    
    name_logging, name_model, name_writer, name = get_experiment_name(
        model_name=model_name, output_dir=output_dir
    )
    total_loss = 0.0
    rec_loss = 0.0
    clus_loss = 0.0
    num_epochs = num_epochs
    model.train()
    threshold = 0.0
    losses = []
    test_acc = []
    best_acc = 0.0
    best_loss = 1000000000
    niter = 1
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    logging.basicConfig(filename=name_logging, level=logging.INFO)
    logging.info(f"Started training model {name} at {now}.")
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        print("Training epoch {}".format(epoch))
        model.train()
        batches = []

        for i, data in enumerate(dataloader, 0):
            image, rotated_jitter_translated, rotated_jitter_translated2, serial_number = data
            x_i = rotated_jitter_translated.to(device)
            x_j = rotated_jitter_translated2.to(device)

            # ===================forward=====================
            with torch.set_grad_enabled(True):
                h_i, h_j, z_i, z_j = model(x_i.permute(0, 2, 1), x_j.permute(0, 2, 1))
                optimizer.zero_grad()
                loss = criterion(z_i, z_j)
                # ===================backward====================
                loss.backward()
                optimizer.step()

            running_loss += loss.detach().item() / batch_size
            batch_num += 1
            niter += 1

            lr = np.asarray(optimizer.param_groups[0]["lr"])

            if i % 10 == 0:
                print(
                    "[%d/%d][%d/%d]\tLossTot: %.2f "
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        loss.detach().item() / batch_size
                    )
                )
                logging.info(
                    f"[{epoch}/{num_epochs}]"
                    f"[{i}/{len(dataloader)}]"
                    f"LossTot: {loss.detach().item() / batch_size}"
                )

        # ===================log========================
        total_loss = running_loss / len(dataloader)
        if total_loss < best_loss:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "loss": total_loss,
            }
            best_loss = total_loss
            create_dir_if_not_exist(output_path)
            print(
                "Saving model to:"
                + name_net
                + ".pt"
                + " with loss = {}".format(total_loss)
                + " at epoch {}".format(epoch)
            )
            logging.info(f"Saving model to {name_model} with loss = {best_loss}.")
            torch.save(checkpoint, name_model)
            print("epoch [{}/{}], loss:{}".format(epoch + 1, num_epochs, total_loss))

        print(
            "epoch [{}/{}], loss:{:.4f}, Rec loss:{:.4f}".format(
                epoch + 1, num_epochs, total_loss, total_loss
            )
        )
        
