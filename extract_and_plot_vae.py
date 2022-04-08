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
    PointCloudDatasetAllBothNotSpec2DRotation1024,
    PointCloudDatasetAllBothKLDivergranceRotation1024
)
import argparse
import os
from tearing.folding_decoder import FoldingNetBasicDecoder
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.express as px


def extract_and_save(mod, dl, save_to, dataframe):
    print("Extracting All")
    from tqdm import tqdm
    criterion = ChamferLoss()
    inputs_test = []
    outputs_test = []
    embeddings_test = []
    labels_test = []
    serial_numbers = []
    mus = []
    log_vars = []
    mod.eval()
    loss = 0
    for data in tqdm(dl):
        with torch.no_grad():
            pts, lab, serial_num = data

            labels_test.append(lab.detach().numpy())
            inputs = pts.to(device)
            outs, mu, log_var, embeddings, z = mod(inputs.permute(0, 2, 1))
            inputs_test.append(torch.squeeze(inputs).cpu().detach().numpy())
            outputs_test.append(torch.squeeze(outs).cpu().detach().numpy())
            embeddings_test.append(torch.squeeze(embeddings).cpu().detach().numpy())
            serial_numbers.append(serial_num)
            mus.append(torch.squeeze(mu).cpu().detach().numpy())
            log_vars.append(torch.squeeze(log_var).cpu().detach().numpy())
            loss += criterion(inputs, outs)

    print(loss / len(dl))
    folding_data = pd.DataFrame(np.asarray(embeddings_test))
    folding_data["serialNumber"] = np.asarray(serial_numbers)
    all_data = pd.read_csv(dataframe)
    all_data_labels = all_data[
        [
            "serialNumber",
            "Treatment",
            "Proximal",
            "nucleusCoverslipDistance",
            "erkRatio",
            "erkIntensityNucleus",
            "erkIntensityCell",
        ]
    ]
    folding_data_new = folding_data.join(
        all_data_labels.set_index("serialNumber"), on="serialNumber"
    )

    folding_data_new.to_csv(
        save_to
    )

    points = outputs_test[100]

    data = pd.DataFrame(points, columns=['x', 'y', 'z'])
    fig = px.scatter_3d(data, x="x", y="y", z='z',
                        color_discrete_sequence=['red']
                        )

    fig.update_traces(marker=dict(size=2),
                      selector=dict(mode='markers'))
    fig.update_layout(
        width=600,
        height=600
    )
    fig.update_layout(scene_aspectmode='data')
    plt.show()

    points = inputs_test[100]

    data = pd.DataFrame(points, columns=['x', 'y', 'z'])
    fig = px.scatter_3d(data, x="x", y="y", z='z',
                        color_discrete_sequence=['red']
                        )

    fig.update_traces(marker=dict(size=2),
                      selector=dict(mode='markers'))
    fig.update_layout(
        width=600,
        height=600
    )
    fig.update_layout(scene_aspectmode='data')
    plt.show()

    return inputs_test, outputs_test, embeddings_test, serial_numbers, mus, log_vars


class MLPVariationalAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, latent_dim=8):
        super(MLPVariationalAutoencoder, self).__init__()
        self.encoder = encoder
        self.enc_mu = nn.Linear(50, latent_dim)
        self.enc_log_var = nn.Linear(50, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 50)
        self.decoder = decoder

    def reparametrize(self, mus, log_vars):
        sigma = torch.exp(0.5*log_vars)
        z = torch.randn(size=(mus.size(0), mus.size(1)))
        z = z.type_as(mus)
        return mus + sigma * z

    def forward(self, x):
        embeddings = self.encoder(x)
        mu = self.enc_mu(embeddings)
        log_var = self.enc_log_var(embeddings)
        z = self.reparametrize(mu, log_var)
        upsamples = self.fc_dec(z)
        outs, grid = self.decoder(upsamples)
        return outs, mu, log_var, embeddings, z


def latent_loss(mu, log_var):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    return kld_loss


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
        default="/home/mvries/Documents/GitHub/FoldingNetNew/nets/"
                "FoldingNetNew_50feats_planeshape_foldingdecoder_trainallTrue_centringonlyTrue_train_bothTrue_003.pt",
        type=str,
    )
    parser.add_argument(
        "--full_checkpoint_path",
        default="/home/mvries/Documents/GitHub/pointMLP-pytorch/"
                "pointmlpelite_foldingTearingVersion_autoencoder_allparams1024VAE.pt",
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

    name_net = output_path + "pointmlpelite_foldingTearingVersion_autoencoder_allparams1024VAE"
    print("==> Building encoder...")
    net = pointMLPElite(num_classes=15)
    device = "cuda"
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    new_embedding = nn.Linear(in_features=256, out_features=50, bias=True)
    net.module.classifier[8] = new_embedding
    net.module.classifier[8].weight.requires_grad = True
    net.module.classifier[8].bias.requires_grad = True
    print(net.module.classifier)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("==> Building decoder...")
    decoder = FoldingNetBasicDecoder(num_features=50, num_clusters=10)

    model = MLPVariationalAutoencoder(encoder=net.module, decoder=decoder).cuda()

    checkpoint = torch.load(full_checkpoint_path)

    model.load_state_dict(torch.load(full_checkpoint_path)['model_state_dict'])

    dataset = PointCloudDatasetAllBothKLDivergranceRotation1024(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        centring_only=True,
        cell_component="cell",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    save_to_path = "pointmlpelite_foldingTearingVersion_autoencoder_allparams1024VAE_50_both_all_centring.csv"
    inputs, outputs, embeddings, serials, muss, log_varss = extract_and_save(model, 
                                                                             dataloader, 
                                                                             save_to_path, 
                                                                             df
                                                                             )
    