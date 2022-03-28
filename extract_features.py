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
from dataset import PointCloudDatasetAllBoth, PointCloudDatasetAllBothNotSpec
import argparse
import os
from tearing.folding_decoder import FoldingNetBasicDecoder


class MLPAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(MLPAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        embedding = self.encoder(x)
        output, grid = self.decoder(embedding)
        return output, embedding, grid


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
    parser.add_argument("--pmlp_folding_ckpt_path", default="/run/user/1128299809/gvfs/smb-share:server=rds.icr.ac.uk,share=data/DBI/DUDBI/DYNCESYS/mvries/ResultsAlma/pointMLP-pytorch/pointmlp_foldingtearingVersion_autoencoder.pt", type=str)


    args = parser.parse_args()
    df = args.dataframe_path
    root_dir = args.dataset_path
    output_path = args.output_path
    num_epochs = args.num_epochs
    pmlp_ckpt_path = args.pmlp_ckpt_path
    fold_ckpt_path = args.fold_ckpt_path
    pmlp_folding_ckpt_path = args.pmlp_folding_ckpt_path

    net = pointMLP()
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

    to_eval = (
            "ReconstructionNet"
            + "("
            + "'{0}'".format("dgcnn_cls")
            + ", num_clusters=5, num_features = 50, shape='plane')"
    )
    decoder = eval(to_eval)
    
    model = MLPAutoencoder(encoder=net.module, decoder=decoder.decoder).cuda()
    checkpoint = torch.load(pmlp_folding_ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    data = torch.rand(2, 3, 2048).cuda()
    print("===> testing pointMLP ...")
    out, embedding, _ = model(data)
    print(out.shape)
    print(embedding.shape)

    batch_size = 16
    learning_rate = 0.00001
    dataset = PointCloudDatasetAllBothNotSpec(
        df,
        root_dir,
        transform=None,
        img_size=400,
        target_transform=True,
        centring_only=True,
        cell_component="cell",
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    from tqdm import tqdm

    inputs_test = []
    outputs_test = []
    features_test = []
    embeddings_test = []
    clusterings_test = []
    labels_test = []
    serial_numbers = []
    model.eval()
    for data in tqdm(dataloader):
        with torch.no_grad():
            pts, lab, _, serial_num = data

            labels_test.append(lab.detach().numpy())
            inputs = pts.to(device)
            output, embedding, _ = model(inputs.permute(0, 2, 1))
            inputs_test.append(torch.squeeze(inputs).cpu().detach().numpy())
            outputs_test.append(torch.squeeze(output).cpu().detach().numpy())
            embeddings_test.append(torch.squeeze(embedding).cpu().detach().numpy())
            serial_numbers.append(serial_num)

    folding_data = pd.DataFrame(np.asarray(embeddings_test))
    folding_data['serialNumber'] = np.asarray(serial_numbers)
    all_data = pd.read_csv(df)
    all_data_labels = all_data[['serialNumber', 'Treatment', 'Proximal', 'nucleusCoverslipDistance', 'erkRatio',
                                'erkIntensityNucleus', 'erkIntensityCell']]
    folding_data_new = folding_data.join(all_data_labels.set_index('serialNumber'), on='serialNumber')

    folding_data_new.to_csv('/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/pointmlpfolding_50_cellandcnuc_all_centring.csv')