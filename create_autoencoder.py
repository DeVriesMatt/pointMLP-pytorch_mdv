import torch
from torch import nn
import classification_ModelNet40.models as models
import torch.backends.cudnn as cudnn
from classification_ModelNet40.models import pointMLP

# from cell_dataset import PointCloudDatasetAllBoth
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from foldingnet import ReconstructionNet, ChamferLoss
from dataset import PointCloudDatasetAllBoth
import argparse
import os


class MLPAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MLPAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        embedding = self.encoder(x)
        output, folding1 = self.decoder(embedding)
        return output, embedding, folding1


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
    parser.add_argument(
        "--fold_ckpt_path",
        default="/home/mvries/Documents/GitHub/FoldingNetNew/nets/FoldingNetNew_50feats_planeshape_foldingdecoder_trainallTrue_centringonlyTrue_train_bothTrue_003.pt",
        type=str,
    )

    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    pmlp_ckpt_path = args.pmlp_ckpt_path
    fold_ckpt_path = args.fold_ckpt_path

    name_net = output_path + "pointmlp_folding_autoencoder"
    print("==> Building encoder...")
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
    new_embedding = nn.Linear(in_features=256, out_features=50, bias=True)
    net.module.classifier[8] = new_embedding
    net.module.classifier[8].weight.requires_grad = True
    net.module.classifier[8].bias.requires_grad = True
    print(net.module.classifier)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("==> Building decoder...")
    to_eval = (
        "ReconstructionNet"
        + "("
        + "'{0}'".format("dgcnn_cls")
        + ", num_clusters=5, num_features = 50, shape='plane')"
    )
    decoder = eval(to_eval)
    fold_path = fold_ckpt_path
    state_dict = torch.load(fold_path)
    model_state_dict = state_dict["model_state_dict"]
    decoder.load_state_dict(model_state_dict)
    print(decoder.decoder)

    model = MLPAutoencoder(encoder=net.module, decoder=decoder.decoder).cuda()
    data = torch.rand(2, 3, 2048).cuda()
    print("===> testing pointMLP ...")
    out, embedding, _ = model(data)
    print(out.shape)
    print(embedding.shape)

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

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate * 16 / batch_size,
        betas=(0.9, 0.999),
        weight_decay=1e-6,
    )
    criterion = ChamferLoss()
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
            inputs, labels, _ = data
            inputs = inputs.to(device)

            # ===================forward=====================
            with torch.set_grad_enabled(True):
                output, embedding, _ = model(inputs.permute(0, 2, 1))
                optimizer.zero_grad()
                loss = criterion(inputs, output)
                # ===================backward====================
                loss.backward()
                optimizer.step()

            running_loss += loss.detach().item() / batch_size
            batch_num += 1
            niter += 1

            lr = np.asarray(optimizer.param_groups[0]["lr"])

            if i % 10 == 0:
                print(
                    "[%d/%d][%d/%d]\tLossTot: %.4f\tLossRec: %.4f"
                    % (
                        epoch,
                        num_epochs,
                        i,
                        len(dataloader),
                        loss.detach().item() / batch_size,
                        loss.detach().item() / batch_size,
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
