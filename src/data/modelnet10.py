# project imports
from data.voxels import voxelize_file, read_voxel_array
from data.dataset import IndexedDataset
from data import MODELNET10_DIR


# python packages
import urllib.request
import pandas as pd
import numpy as np
import traceback
import shutil
import json
import time
import csv
import os


def process_modelnet10_model_dir(model_dir, voxels_dim=32):
    """
    1. Voxelize
    2. Index

    Args:
        root: str, dir that contains models
    """
    voxel_files = list()
    for model_file in os.listdir(model_dir):
        if '.off' not in model_file:
            continue
        model_file = os.path.join(model_dir, model_file)
        for x in range(4):
            for z in range(4):
                voxel_file = voxelize_file(model_file, ext='off', dest_dir=None, size=voxels_dim, verbose=True,
                                           num_rotx=x, num_rotz=z, binvox_suffix='_{}_x{}_z{}'.format(voxels_dim, x, z))
                voxel_file_info = [os.path.basename(model_file), os.path.basename(voxel_file), voxels_dim, x, z]
                voxel_files.append(voxel_file_info)
    return voxel_files


def make_modelnet10_index(root, output, categories=None):
    """
    Args:
        root: str, root of ModelNet10 directory
    """
    modelnet_categories = os.listdir(root)
    if categories:
        # filter to only teh categories we want
        modelnet_categories = [cat for cat in modelnet_categories if cat in categories]
    if 'README.txt' in modelnet_categories:
        modelnet_categories.remove('README.txt')
    models = list()
    for category in modelnet_categories:
        # test dir
        test_dir = os.path.join(root, category, 'test')
        models_processed = process_modelnet10_model_dir(test_dir)
        models_processed = [[category, 'test'] + m for m in models_processed]
        models += models_processed
        # train dir
        train_dir = os.path.join(root, category, 'train')
        models_processed = process_modelnet10_model_dir(train_dir)
        models_processed = [[category, 'train'] + m for m in models_processed]
        models += models_processed
    # output csv
    with open(output, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['category', 'dataset', 'model', 'binvox', 'dimension', 'x_rotations', 'z_rotations'])
        for row in models:
            csvwriter.writerow(row)
    print('ModelNet10 csv written to {}'.format(output))
    return
        

class ModelNet10(IndexedDataset):
    
    ID_COL = 'model'
    
    def __init__(self, df, index, pctile):
        super().__init__(df, index, pctile)
        
    def get_random_voxels(self, voxels_dim):
        random_sample = self.df.sample(n=1).iloc[0]
        voxels = self.get_voxels(random_sample['category'], random_sample['dataset'], random_sample['binvox'])
        return voxels

    def get_voxels(self, category, dataset, voxel_file, verbose=False, shape=None):
        # construct path
        vox_path = os.path.join(MODELNET10_DIR, category, dataset, voxel_file)
        if verbose:
            print('Voxel path: {}'.format(vox_path))
        # read in voxels
        vox = read_voxel_array(vox_path)
        # convert from bool True/False to float 1/0 (tf likes floats)
        vox_data = vox.data.astype(np.float32)
        if shape:
            vox_data = np.reshape(vox_data, shape)
        return vox_data

    def voxels_batchmaker(self, batch_size, voxels_dim, set_filter=None, verbose=False, pad=False):
        """
        Args:
            batch_size: int
            set_filter: str, "train" or "test" or None; returns specified dataset or both
            verbose: bool, extra debug prints
            pad: bool, pads final batch with all-zero examples or skips remainder
        """
        batch = list()
        if set_filter in ['train', 'test']:
            df_slicer = self.df.dataset == set_filter
            if verbose:
                print('Creating dataset split for "{}"'.format(set_filter))
        else:
            df_slicer = self.df == self.df
        for i, (index, row) in enumerate(self.df[df_slicer].iterrows()):
            vox_data = self.get_voxels(row['category'], row['dataset'], row['binvox'], shape=[voxels_dim, voxels_dim, voxels_dim, 1])
            if vox_data is None:
                continue
            # each element has 1 "channel" aka data point (if RGB color, it would be 3)
            batch.append(vox_data) 
            # yield batch if ready; else continue
            if (i+1) % batch_size == 0:
                yield np.asarray(batch)
                batch = list()
            if pad and (i+1 == len(self.df)):
                # if we've reached the end of the loop and do not have enough for a batch
                # (and if user said pad=True)
                # then pad the batch and yield
                for x in range(batch_size - len(batch)):
                    batch.append(np.zeros([voxels_dim, voxels_dim, voxels_dim, 1]))
                yield np.asarray(batch)
        return

        
    def __repr__(self):
        return '<ModelNet10()>'
