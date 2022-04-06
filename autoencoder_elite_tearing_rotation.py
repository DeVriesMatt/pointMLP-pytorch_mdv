import torch
from torch import nn
import classification_ModelNet40.models as models
import torch.backends.cudnn as cudnn
from classification_ScanObjectNN.models import pointMLPElite

# from cell_dataset import PointCloudDatasetAllBoth
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from foldingnet import ReconstructionNet, ChamferLoss
from angle_loss import AngleLoss
from dataset import (
    PointCloudDatasetAllBoth,
    PointCloudDatasetAllBothNotSpec,
    PointCloudDatasetAllBothNotSpec1024,
    PointCloudDatasetAllBothNotSpecRotation,
    PointCloudDatasetAllBothNotSpecRotation1024,
    PointCloudDatasetAllBothNotSpec2DRotation1024
)
import argparse
import os
from tearing.folding_decoder import FoldingNetBasicDecoder


class MLPAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MLPAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        embeddings = self.encoder(x)
        predicted_angels = embeddings[:, :3]
        outs, grid = self.decoder(embeddings)
        return outs, embeddings, grid, predicted_angels


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pointmlp-foldingnet")
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
                "pointmlpelite_foldingTearingVersion_autoencoder_allparams.pt",
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

    name_net = output_path + "pointmlpelite_foldingTearingVersion_autoencoder_allparams1024Rotation"
    print("==> Building encoder...")
    net = pointMLPElite(num_classes=15)
    device = "cuda"
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # checkpoint_path = pmlp_ckpt_path
    # checkpoint = torch.load(checkpoint_path)
    # net.load_state_dict(checkpoint["net"])
    # for param in net.module.parameters():
    #     param.requires_grad = False
    new_embedding = nn.Linear(in_features=256, out_features=50, bias=True)
    net.module.classifier[8] = new_embedding
    net.module.classifier[8].weight.requires_grad = True
    net.module.classifier[8].bias.requires_grad = True
    print(net.module.classifier)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("==> Building decoder...")
    decoder = FoldingNetBasicDecoder(num_features=50, num_clusters=10)

    model = MLPAutoencoder(encoder=net.module, decoder=decoder).cuda()

    checkpoint = torch.load(full_checkpoint_path)
    # model_dict = model.state_dict()  # load parameters from pre-trained FoldingNet
    # for k in checkpoint["model_state_dict"]:
    #     if "decoder" in k:
    #         if k in model_dict:
    #             model_dict[k] = checkpoint["model_state_dict"][k]
    #             print("    Found weight: " + k)
    #         elif k.replace("folding1", "folding") in model_dict:
    #             model_dict[k.replace("folding1", "folding")] = checkpoint[
    #                 "model_state_dict"
    #             ][k]
    #             print("    Found weight: " + k)
    model.load_state_dict(torch.load(full_checkpoint_path)['model_state_dict'])

    data = torch.rand(2, 3, 1024).cuda()
    print("===> testing pointMLP ...")
    out, embedding, _, pred_a = model(data)
    print(out.shape)
    print(embedding.shape)

    batch_size = 16
    learning_rate = 0.0000001
    dataset = PointCloudDatasetAllBothNotSpecRotation1024(
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
        lr=learning_rate * 16 / batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-8,
    )
    criterion = ChamferLoss()
    criterion_rot_a = AngleLoss()
    criterion_rot_b = AngleLoss()
    criterion_rot_c = AngleLoss()
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
    for epoch in range(num_epochs):
        batch_num = 1
        running_loss = 0.0
        print("Training epoch {}".format(epoch))
        model.train()
        batches = []

        for i, data in enumerate(dataloader, 0):
            image, rotated_image, angles, serial_number = data
            inputs = image.to(device)
            rotated_inputs = rotated_image.to(device)
            angles = angles.to(device)

            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output, embedding, _, pred_angles = model(rotated_inputs.permute(0, 2, 1))
                optimizer.zero_grad() 
                loss_rec = criterion(inputs, output)
                alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
                alpha_hat, beta_hat, gamma_hat = pred_angles[:, 0], pred_angles[:, 1], pred_angles[:, 2]
                # print(alpha, alpha_hat)
                loss_rot_alpha = criterion_rot_a(alpha, alpha_hat)
                loss_rot_beta = criterion_rot_b(beta, beta_hat)
                loss_rot_gamma = criterion_rot_c(gamma, gamma_hat)
                # ===================backward====================
                loss = loss_rec + 0.1*(loss_rot_alpha + loss_rot_beta + loss_rot_gamma)
                loss.backward()
                optimizer.step()

            running_loss += loss.detach().item() / batch_size
            batch_num += 1
            niter += 1

            lr = np.asarray(optimizer.param_groups[0]["lr"])

            if i % 10 == 0:
                print(
                    "[%d/%d][%d/%d]\tLossTot: %.2f\tLossRec: %.2f \tLossAlpha: %.2f \tLossBeta: %.2f"
                    "\tLossGamma: %.4f"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        loss.detach().item() / batch_size,
                        loss_rec.detach().item() / batch_size,
                        loss_rot_alpha.detach().item(),
                        loss_rot_beta.detach().item(),
                        loss_rot_gamma.detach().item()
                    )
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
            torch.save(checkpoint, name_net + ".pt")
            print("epoch [{}/{}], loss:{}".format(epoch + 1, num_epochs, total_loss))

        print(
            "epoch [{}/{}], loss:{:.4f}, Rec loss:{:.4f}".format(
                epoch + 1, num_epochs, total_loss, total_loss
            )
        )
