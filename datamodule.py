import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

import h5py
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import (BatchSampler, SequentialSampler,
                                      WeightedRandomSampler)

from utils import WeightForBalancedDataset, return_df_from_csv

logger = logging.getLogger(__name__)


class BalancedBatchSampler(BatchSampler):

    def __init__(self, csv, n_classes, n_samples):
        filelist = return_df_from_csv(csv)
        self.labels_list = list(filelist['cyto_diag_groups'])
        self.labels = torch.ByteTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0]
            for label in self.labels_set
        }
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = filelist
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        while self.used_label_indices_count[3] + self.batch_size < len(self.label_to_indices[3]):
            classes = np.random.choice(
                self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(
                    self.label_to_indices[class_][self.used_label_indices_count[class_]:
                                                  self.used_label_indices_count[class_]+self.n_samples]
                )
                self.used_label_indices_count[class_] += self.n_samples

                if class_ == 1 or class_ == 2:
                    if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                        np.random.shuffle(self.label_to_indices[class_])
                        self.used_label_indices_count[class_] = 0
            yield indices

    def __len__(self):
        return len(self.dataset) // self.batch_size


class FeaturizedDataset(Dataset):

    def __init__(
        self,
        splits_csv_path: Union[Path, str],
        features_dir: Union[Path, str]
    ) -> None:
        super().__init__()

        self.features_list = return_df_from_csv(splits_csv_path)
        self.features_dir = features_dir

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, index):
        row = self.features_list.iloc[index]
        X_fname, label = row['original_filename'], row['cyto_diag_groups']

        filename = str(X_fname)
        filename = f'{filename[:-4]}_bag_features.h5'
        X_dir = f'{self.features_dir}/{filename}'

        with h5py.File(X_dir, 'r') as f:
            X = np.asarray(
                f['features'],
                dtype=np.float32
            )
            X = torch.from_numpy(X)

            X_coords = np.asarray(
                f['coords']
            )

        if label == 1:
            label = int(0)
        elif label == 3:
            label = int(1)
        label = torch.from_numpy(np.array(label))

        sample = {
            'X': X,
            'y': label,
            'coords': X_coords,
            'filename': filename
        }

        return sample


class FeaturizedDataModule(LightningDataModule):
    def __init__(
        self,
        splits_root_dir: str,
        features_root_dir: str,
        fold_num: int,
        batch_size: int,
        num_workers: int,
    ) -> None:

        super().__init__()
        self.splits_root_dir = splits_root_dir
        self.features_root_dir = features_root_dir
        self.fold_num = fold_num

        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        train_fold = f'{self.splits_root_dir}/fold_{self.fold_num}_train.tsv'
        val_fold = f'{self.splits_root_dir}/fold_{self.fold_num}_val.tsv'
        test_fold = f'{self.splits_root_dir}/fold_{self.fold_num}_test.tsv'

        if stage == "fit" or stage is None:
            self.sampling_weights = WeightForBalancedDataset(
                splits_csv_file=train_fold,
                num_classes=2
            )

            self.train_dset = FeaturizedDataset(
                splits_csv_path=train_fold,
                features_dir=self.features_root_dir
            )
            self.val_dset = FeaturizedDataset(
                splits_csv_path=val_fold,
                features_dir=self.features_root_dir
            )

        if stage == "test":
            self.test_dset = FeaturizedDataset(
                splits_csv_path=test_fold,
                features_dir=self.features_root_dir
            )

    def train_dataloader(self):
        train_sampler = WeightedRandomSampler(
            self.sampling_weights,
            len(self.sampling_weights),
            replacement=True
        )
        return DataLoader(
            dataset=self.train_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            sampler=train_sampler,
            collate_fn=self.collate_MIL
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            sampler=SequentialSampler(self.val_dset),
            collate_fn=self.collate_MIL
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            sampler=SequentialSampler(self.test_dset),
            collate_fn=self.collate_MIL
        )

    @staticmethod
    def collate_MIL(batch):
        img = torch.cat([item["X"] for item in batch], dim=0)
        label = torch.LongTensor([item["y"] for item in batch])
        coords = [item["coords"] for item in batch]
        filename = [item["filename"] for item in batch]
        return {
            "img": img,
            "label": label,
            "coords": coords,
            "filename": filename
        }
