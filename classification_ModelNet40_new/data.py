import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))

def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join('/home/mvries/CurveNet', 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


def random_scale(point_data, scale_low=0.8, scale_high=1.2):
    """Randomly scale the point cloud. Scale is per point cloud.
    Input:
        Nx3 array, original batch of point clouds
    Return:
        Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    scaled_pointcloud = np.multiply(point_data, scale).astype("float32")
    return scaled_pointcloud


def translate_pointcloud(pointcloud):
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(pointcloud, shift).astype("float32")
    return translated_pointcloud


class Intra3D(Dataset):
    def __init__(
        self,
        points_dir="/data/scratch/DBI/DUDBI/DYNCESYS/mvries/Datasets/IntrA/",
        train_mode="train",
        cls_state=True,
        npoints=1024,
        data_aug=True,
        choice=1,
    ):
        self.npoints = npoints  # 2048 pts
        self.data_augmentation = data_aug
        self.datapath = []
        self.label = {}
        self.cls_state = cls_state
        self.train_mode = train_mode
        fold_csv = pd.read_csv(points_dir + f"folds/fold_{choice}.csv")

        if self.cls_state:
            self.label[0] = glob.glob(
                points_dir + "generated/vessel/ad/" + "*.ad"
            )  # label 0: healthy; 1694 files; negSplit
            self.label[1] = glob.glob(
                points_dir + "generated/aneurysm/ad/" + "*.ad"
            ) + glob.glob(
                points_dir + "annotated/ad/" + "*.ad"
            )  # label 1: unhealthy; 331 files

            train_set = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv[fold_csv["Split"] == "train"]["Path"].tolist()
            ]
            val_set = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv[fold_csv["Split"] == "validation"]["Path"].tolist()
            ]
            train_set = train_set + val_set
            test_set = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv[fold_csv["Split"] == "test"]["Path"].tolist()
            ]
        else:

            annotated = glob.glob(points_dir + "annotated/ad/" + "*.ad")
            train_set = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv[fold_csv["Split"] == "train"]["Path"].tolist()
                if points_dir + i.split("IntrA/")[-1] in annotated
            ]
            val_set = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv[fold_csv["Split"] == "validation"]["Path"].tolist()
                if points_dir + i.split("IntrA/")[-1] in annotated
            ]
            train_set = train_set + val_set
            test_set = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv[fold_csv["Split"] == "test"]["Path"].tolist()
                if points_dir + i.split("IntrA/")[-1] in annotated
            ]
            non_annotated = [
                points_dir + i.split("IntrA/")[-1]
                for i in fold_csv["Path"].tolist()
                if points_dir + i.split("IntrA/")[-1] not in annotated
            ]

            # train_set = [i.split("/")[-1] for i in fold_csv[fold_csv['Split'] == 'train']['Path'].tolist()]
            # val_set = [i.split("/")[-1] for i in fold_csv[fold_csv['Split'] == 'validation']['Path'].tolist()]
            # train_set = train_set + val_set
            # test_set = [i.split("/")[-1] for i in fold_csv[fold_csv['Split'] == 'test']['Path'].tolist()]

        if self.train_mode == "train":
            self.datapath = train_set

        elif self.train_mode == "test":
            self.datapath = test_set

        elif self.train_mode == "all":
            self.datapath = train_set + test_set

        elif self.train_mode == "interpret":
            self.datapath = annotated

        elif self.train_mode == "non_annotated":
            self.datapath = non_annotated

        else:
            print("Error")
            raise Exception("training mode invalid")

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        curr_file = self.datapath[index]
        cls = None
        if self.cls_state:

            if curr_file in self.label[0]:
                cls = torch.from_numpy(np.array([0]).astype(np.int64))

            elif curr_file in self.label[1]:
                cls = torch.from_numpy(np.array([1]).astype(np.int64))
            else:
                print("Error found!!!")
                exit(-1)

        point_set = np.loadtxt(curr_file)[:, :-1].astype(
            np.float32
        )  # [x, y, z, norm_x, norm_y, norm_z]
        seg = np.loadtxt(curr_file)[:, -1].astype(np.int64)  # [seg_label]
        seg[np.where(seg == 2)] = 1  # making boundary lines (label 2) to A. (label 1)

        # random choice
        if point_set.shape[0] < self.npoints:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.npoints, replace=False)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # normalization to unit ball
        point_set[:, :3] = point_set[:, :3] - np.mean(
            point_set[:, :3], axis=0
        )  # x, y, z
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)), 0)
        point_set[:, :3] = point_set[:, :3] / dist

        # data augmentation
        if self.data_augmentation:
            if self.train_mode == "train":
                point_set[:, :3] = random_scale(point_set[:, :3])
                point_set[:, :3] = translate_pointcloud(point_set[:, :3])
            if self.train_mode == "test":
                point_set[:, :3] = point_set[:, :3]

        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        return (point_set, seg, np.array([1])) if not self.cls_state else (point_set, cls)



if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = ModelNet40(partition='train', num_points=1024)
    test_set = ModelNet40(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")



