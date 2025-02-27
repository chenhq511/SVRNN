import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr

def SplitSeq(data, seq_len, stride):
    T, C, H, W = data.shape
    N = max(1, (T - seq_len) // stride + 1)
    shape = (N, seq_len, C, H, W)
    strides = (data.strides[0] * stride, *data.strides)
    split_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return split_data

class CMEMS(Dataset):
    def __init__(self, args, data_type):
        super(CMEMS, self).__init__()
        self.num_frames_input = args.num_frames_input
        self.num_frames_output = args.num_frames_output
        self.num_frames_total = args.num_frames_input + args.num_frames_output

        if data_type == 'train' or data_type == 'val':
            self.data_dir = args.train_data_dir
        else:
            self.data_dir = args.test_data_dir

        try:
            self.datas = xr.open_dataset(self.data_dir, engine='netcdf4')
        except FileNotFoundError as e:
            print(f"File not found: {self.data_dir}")
            raise e

        self.datas = self.datas[list(self.datas.data_vars)[0]]
        self.datas = self.datas[:, :, 1:, 1:]   #(72,72)
        self.mask =self.datas.notnull().astype(int)
        self.datas = self.datas.values
        self.max_nan = np.nanmax(self.datas)
        self.min_nan = np.nanmin(self.datas)
        self.datas = (self.datas - self.min_nan) / (self.max_nan - self.min_nan)
        self.datas = np.nan_to_num(self.datas, nan=0)
        self.datas = SplitSeq(self.datas, args.frames_per_subsample,args.next_step)

        if data_type == 'train' or data_type == 'val':
            np.random.seed(42)
            full_indices = np.arange(len(self.datas))
            np.random.shuffle(full_indices)
            if data_type == 'train':
                train_indices = full_indices[:int(len(full_indices) * 0.8)]
                self.datas = self.datas[train_indices]
            else:
                val_indices = full_indices[int(len(full_indices) * 0.8):]
                self.datas = self.datas[val_indices]
        self.datas = torch.from_numpy(self.datas).permute(0, 1, 3, 4, 2)

    def find_max_min(self):
        return self.max_nan, self.min_nan

    def mask_data(self):
        return self.mask[0, 0, :, :]
    def __getitem__(self, idx):
        inputs = self.datas[idx, :self.num_frames_input]
        targets = self.datas[idx, self.num_frames_input: self.num_frames_total]
        return inputs, targets

    def __len__(self):
        return self.datas.shape[0]


if __name__ == '__main__':
    # -----------------------------------------------------------------------------
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_frames_input',      default=12)
    parser.add_argument('--num_frames_output',     default=12)
    parser.add_argument('--frames_per_subsample',  default=24)
    parser.add_argument('--next_step',      default=2)
    parser.add_argument('--train_data_dir', default='./data/thetao_train.nc')
    parser.add_argument('--val_data_dir',   default='./data/thetao_test.nc')

    args = parser.parse_args()



