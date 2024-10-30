import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
from tqdm import tqdm


class GetDataset(Dataset):
    def __init__(self, upper_path, surfa_path, start_year=2002,
                 end_year=2022, preci=False, time_step=1, samp_step=1, in_length=2, out_length=1, ):
        self.upper_path = upper_path
        self.surfa_path = surfa_path
        self.time_step = time_step
        self.samp_step = samp_step
        self.in_length = in_length
        self.out_length = out_length
        self.start_year = start_year
        self.end_year = end_year
        self.preci = preci
        self._get_files_stats()
        self.upper_mean = np.mean(np.load('/mnt/data/cra_h5/upper/stat_np/upper_mean_22_79.npy'), axis=0).squeeze(0)
        self.upper_std = np.sqrt(
            np.mean(np.load('/mnt/data/cra_h5/upper/stat_np/upper_var_22_79.npy'), axis=0).squeeze(0))
        self.surface_mean = np.mean(np.load('/mnt/data/cra_h5/surface_linear/stat_np/surface_mean_22_79.npy'),
                                    axis=0).squeeze(0)
        self.surface_std = np.sqrt(
            np.mean(np.load('/mnt/data/cra_h5/surface_linear/stat_np/surface_var_22_79.npy'), axis=0).squeeze(0))

        self.add_data = [self._addition_feature(1458, year) for year in range(start_year, end_year)]
    def _get_files_stats(self):
        self.upper_files = glob.glob(self.upper_path + "/*.h5")
        self.surfa_filea = glob.glob(self.surfa_path + "/*.h5")
        self.upper_files.sort()
        self.surfa_filea.sort()
        if self.start_year:
            self.upper_files = [
                file_path for file_path in self.upper_files
                if self.start_year <= int(file_path.split('/')[-1].split('.')[0]) <= self.end_year
            ]
            self.surfa_filea = [
                file_path for file_path in self.surfa_filea
                if self.start_year <= int(file_path.split('/')[-1].split('.')[0]) <= self.end_year
            ]
        assert len(self.upper_files) == len(self.surfa_filea)
        self.n_years = len(self.upper_files)
        self.n_samples_per_year = (1460 - (self.in_length + self.out_length) * self.time_step + 1) // self.samp_step
        self.n_samples_total = self.n_years * self.n_samples_per_year

        self.ufiles = [None for _ in range(self.n_years)]
        self.sfiles = [None for _ in range(self.n_years)]

        self.ufiles = {year_idx: h5py.File(self.upper_files[year_idx], 'r', libver='latest', swmr=True) for year_idx in
                       range(self.n_years)}
        self.sfiles = {year_idx: h5py.File(self.surfa_filea[year_idx], 'r', libver='latest', swmr=True) for year_idx in
                       range(self.n_years)}

    def _addition_feature(self,  n_steps, st_year):
        import pandas as pd
        # Define some constants
        SEC_PER_HOUR = 3600
        HOUR_PER_DAY = 24
        SEC_PER_DAY = SEC_PER_HOUR * HOUR_PER_DAY
        AVG_DAY_PER_YEAR = 365.24219

        # Generate a date range starting from st_year, with n_steps steps, assuming each step is 6 hours
        dates = pd.date_range(start=st_year, periods=n_steps + 2, freq='6h')
        seconds_since_epoch = (dates - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        seconds_since_epoch = np.expand_dims(seconds_since_epoch, axis=0)

        # Calculate year progress
        def get_year_progress(seconds_since_epoch):
            years_since_epoch = seconds_since_epoch / SEC_PER_DAY / AVG_DAY_PER_YEAR
            year_progress = np.mod(years_since_epoch, 1.0).astype(np.float32)
            return year_progress

        year_progress = get_year_progress(seconds_since_epoch)

        # Calculate day progress
        def get_day_progress(seconds_since_epoch, longitude):
            day_progress = (np.mod(seconds_since_epoch, SEC_PER_DAY) / SEC_PER_DAY)
            # Offset the day progress to the longitude of each point on Earth.
            longitude_offsets = np.deg2rad(longitude) / (2 * np.pi)
            day_progress = np.mod(
                day_progress[..., np.newaxis] + longitude_offsets, 1.0
            )
            return day_progress.astype(np.float32)

        longitude = np.linspace(0, 359.75, 1440)

        day_progress = get_day_progress(seconds_since_epoch, longitude)
        return {
            "year_progress_sin": np.sin(year_progress * 2 * np.pi).squeeze(0),
            "year_progress_cos": np.cos(year_progress * 2 * np.pi).squeeze(0),
            "day_progress_sin": np.sin(day_progress * 2 * np.pi).squeeze(0),
            "day_progress_cos": np.cos(day_progress * 2 * np.pi).squeeze(0)
        }
    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year we are on
        local_idx = int(
            global_idx % self.n_samples_per_year)  # which sample in that year we are on - determines indices for centering


        start_i = local_idx * self.samp_step
        end_i = local_idx * self.samp_step + (self.in_length + self.out_length) * self.time_step
        upper_data = self.ufiles[year_idx]['upper'][start_i: end_i: self.time_step]
        upper_data = (upper_data - self.upper_mean) / self.upper_std

        surfa_data = self.sfiles[year_idx]['surface'][start_i: end_i: self.time_step]
        surfa_data = (surfa_data - self.surface_mean) / self.surface_std
        if not self.preci:
            surfa_data = surfa_data[:, :-1]

        upper_data = torch.as_tensor(upper_data, dtype=torch.float32)
        surfa_data = torch.as_tensor(surfa_data, dtype=torch.float32)


        input_data_upper = torch.stack((upper_data[0], upper_data[1]), axis=0).permute(-2, -1, 0, 1, 2)
        input_data_surface = torch.stack((surfa_data[0], surfa_data[1]), axis=0).permute(-2, -1, 0, 1)

        input_data_upper = input_data_upper.reshape(input_data_upper.shape[0] * input_data_upper.shape[1], -1)
        input_data_surface = input_data_surface.reshape(input_data_surface.shape[0] * input_data_surface.shape[1], -1)
        input_data = torch.FloatTensor(np.concatenate((input_data_upper, input_data_surface), axis=-1))

        a = torch.FloatTensor(self.add_data[year_idx]['year_progress_sin'][start_i: end_i: self.time_step]).expand(721,
                                                                                                               1440, -1)
        b = torch.FloatTensor(self.add_data[year_idx]['year_progress_cos'][start_i: end_i: self.time_step]).expand(721,
                                                                                                               1440, -1)
        c = torch.FloatTensor(
            self.add_data[year_idx]['day_progress_sin'][start_i: end_i: self.time_step].reshape(1440, -1)).expand(
            721, -1, -1)
        d = torch.FloatTensor(
            self.add_data[year_idx]['day_progress_cos'][start_i: end_i: self.time_step].reshape(1440, -1)).expand(
            721, -1, -1)
        forcing_data = torch.concat([a, b, c, d], dim=2).reshape(721 * 1440, -1)
        input_data = torch.concat([input_data, forcing_data], dim=1)

        # label
        label_data_upper = torch.FloatTensor(upper_data[2]).permute(-2, -1, 0, 1)
        label_data_surface = torch.FloatTensor(surfa_data[2]).permute(-2, -1, 0)

        label_data_upper = label_data_upper.reshape(label_data_upper.shape[0] * label_data_upper.shape[1], -1)
        label_data_surface = label_data_surface.reshape(label_data_surface.shape[0] * label_data_surface.shape[1], -1)
        label_data = torch.concat((label_data_upper, label_data_surface), dim=-1)

        return input_data, label_data

    def __del__(self):
        # 在析构时关闭所有打开的文件
        for year_idx in self.ufiles:
            self.ufiles[year_idx].close()
        for year_idx in self.sfiles:
            self.sfiles[year_idx].close()


if __name__ == '__main__':
    '''
    数据均为.h5文件
    '''
    upper_path = 'upper_data'
    surface_path = 'surface_data'
    dataset = GetDataset(upper_path=upper_path, surfa_path=surface_path)
    # subset = Subset(dataset, range(720, len(dataset)))

    laoder = DataLoader(dataset, shuffle=False, batch_size=2, num_workers=6)
    for data in tqdm(laoder):
        pass
