import torch
import classification_ModelNet40.models as models
import torch.backends.cudnn as cudnn
from classification_ModelNet40.models import pointMLP
from cell_dataset import PointCloudDatasetAll
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # data = torch.rand(1, 3, 2048).cuda()
    # checkpoint = torch.load('best_checkpoint.pth')
    # print(checkpoint.keys())
    # state_dict = checkpoint['net']
    # print(state_dict)
    # print("===> testing pointMLP ...")
    # model = pointMLP()
    # model.load_state_dict(state_dict).cuda().eval()
    # out = model(data)
    # print(out.shape)

    print("==> Building model..")
    net = pointMLP()
    device = "cuda"
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint_path = "best_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    print(net)
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    net.module.classifier[4].register_forward_hook(get_activation("features"))
    # data = torch.rand(1, 3, 2048).cuda()
    # out = net(data)
    # print(activation['features'].shape)

    df = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/all_cell_data.csv"
    root_dir = "/home/mvries/Documents/Datasets/OPM/SingleCellFromNathan_17122021/"

    dataset = PointCloudDatasetAll(
        df, root_dir, transform=None, img_size=400, target_transform=True
    )

    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader_inf = DataLoader(dataset, batch_size=1, shuffle=False)

    from tqdm import tqdm

    inputs_test = []
    outputs_test = []
    features_test = []
    embeddings_test = []
    clusterings_test = []
    labels_test = []
    serial_numbers = []
    features = []
    for data in tqdm(dataloader_inf):
        with torch.no_grad():
            pts, lab, _, serial_num = data
            inputs = pts.to(device)
            inputs = inputs.permute(0, 2, 1)
            output = net(inputs)
            labels_test.append(torch.squeeze(lab).detach().numpy())
            features.append(torch.squeeze(activation["features"]).cpu().numpy())

    print(activation["features"])
    print(np.array(features).shape)
    df = pd.DataFrame(np.asarray(features))
    df["labels"] = np.asarray(labels_test)
    df.to_csv("cell_features_mlp.csv")
