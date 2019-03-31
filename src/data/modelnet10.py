# project imports
from data.voxels import voxelize_file, read_voxel_array
from data.dataset import IndexedDataset
from data import MODELNET10_DIR


# python packages
import urllib.request
import pandas as pd
import numpy as np
import traceback
import random
import shutil
import json
import time
import csv
import os


# some models either were bad to begin with or did not convert well
# to voxels; we keep track of those here
KNOWN_BAD_MODELNET10 = [
    'table_0412_32_x0_z0.binvox',
    'bathtub_0016_32_x0_z0.binvox',
    'chair_0174_32_x0_z0.binvox',
    'chair_0712_32_x0_z0.binvox',
    'table_0048_32_x0_z0.binvox',
    'monitor_0202_32_x0_z0.binvox',
    'chair_0670_32_x0_z0.binvox',
    'bathtub_0039_32_x0_z0.binvox',
    'table_0438_32_x0_z0.binvox',
    'bathtub_0029_32_x0_z0.binvox',
    'table_0197_32_x0_z0.binvox',
    'desk_0014_32_x0_z0.binvox',
    'bathtub_0076_32_x0_z0.binvox',
    'desk_0179_32_x0_z0.binvox',
    'chair_0242_32_x0_z0.binvox',
    'table_0419_32_x0_z0.binvox',
    'bathtub_0037_32_x0_z0.binvox',
    'monitor_0455_32_x0_z0.binvox',
    'monitor_0122_32_x0_z0.binvox',
    'sofa_0493_32_x0_z0.binvox',
    'monitor_0138_32_x0_z0.binvox',
    'chair_0529_32_x0_z0.binvox',
    'chair_0965_32_x0_z0.binvox',
    'sofa_0214_32_x0_z0.binvox',
    'chair_0699_32_x0_z0.binvox',
    'chair_0265_32_x0_z0.binvox',
    'chair_0352_32_x0_z0.binvox'
    'chair_0584_32_x0_z0.binvox',
    'chair_0550_32_x0_z0.binvox',
    'chair_0254_32_x0_z0.binvox',
    'chair_0875_32_x0_z0.binvox',
    'toilet_0434_32_x0_z0.binvox',
    'toilet_0075_32_x0_z0.binvox', 
    'toilet_0081_32_x0_z0.binvox',
    'chair_0352_32_x0_z0.binvox',
    'chair_0584_32_x0_z0.binvox',
    'chair_0012_32_x0_z0.binvox',
    'chair_0027_32_x0_z0.binvox',
    'chair_0842_32_x0_z0.binvox',
    'chair_0730_32_x0_z0.binvox',
    'chair_0514_32_x0_z0.binvox',
    'chair_0058_32_x0_z0.binvox',
    'chair_0841_32_x0_z0.binvox',
    'chair_0868_32_x0_z0.binvox',
    'chair_0204_32_x0_z0.binvox',
    'chair_0115_32_x0_z0.binvox',
    'chair_0079_32_x0_z0.binvox',
    'chair_0245_32_x0_z0.binvox',
    'chair_0829_32_x0_z0.binvox',
    'chair_0646_32_x0_z0.binvox',
    'chair_0972_32_x0_z0.binvox',
    'chair_0083_32_x0_z0.binvox',
    'bed_0173_32_x0_z0.binvox',
    'bed_0156_32_x0_z0.binvox',
    'bed_0233_32_x0_z0.binvox',
    'chair_0301_32_x0_z0.binvox',
]

# some models are fine but are just not very interesting; we keep track here
KNOWN_NOT_INTERESTING_MODELNET10 = [
    'chair_0658_32_x0_z0.binvox',  # flat rectangle
    'dresser_0116_32_x0_z0.binvox',  # flat box
    'dresser_0078_32_x0_z0.binvox',  # flat box
    'table_0080_32_x0_z0.binvox',  # flat rectangle
    'dresser_0205_32_x0_z0.binvox',  # flat box
    'dresser_0149_32_x0_z0.binvox',  # flat box
    'table_0004_32_x0_z0.binvox',  # just a small L
    'night_stand_0252_32_x0_z0.binvox',  # flat box
    'dresser_0146_32_x0_z0.binvox',  # flat box
    'dresser_0122_32_x0_z0.binvox',  # flat box
    'dresser_0010_32_x0_z0.binvox',  # flat box with a lip on top
    'dresser_0258_32_x0_z0.binvox',  # flat box
    'dresser_0262_32_x0_z0.binvox',  # flat box
    'monitor_0081_32_x0_z0.binvox',  # flat box with the smallest stand
    'chair_0107_32_x0_z0.binvox',  # flat rectangle
    'table_0394_32_x0_z0.binvox',  # flat rectangle
    'monitor_0084_32_x0_z0.binvox',  # flat rectangle
    'dresser_0238_32_x0_z0.binvox',  # flat box
    'dresser_0019_32_x0_z0.binvox',  # flat box
    'monitor_0330_32_x0_z0.binvox',  # flat rectangle
    'monitor_0477_32_x0_z0.binvox',  # flat rectangle
    'monitor_0197_32_x0_z0.binvox',  # flat rectangle
    'monitor_0271_32_x0_z0.binvox',  # flat rectangle
    'monitor_0009_32_x0_z0.binvox',  # flat rectangle
    'monitor_0399_32_x0_z0.binvox',  # flat rectangle
    'chair_0127_32_x0_z0.binvox',  # strange
    'monitor_0318_32_x0_z0.binvox'  # flat rectangle
    'chair_0265_32_x0_z0.binvox',  # too small & no legs
    'sofa_0214_32_x0_z0.binvox',  # mostly flat rectangle
    'monitor_0318_32_x0_z0.binvox',  # flat
    'monitor_0341_32_x0_z0.binvox',  # flat
    'monitor_0332_32_x0_z0.binvox',  # flat
    'monitor_0018_32_x0_z0.binvox',  # flat
    'monitor_0189_32_x0_z0.binvox',  # flat
    'monitor_0558_32_x0_z0.binvox',  # flat, little stand
    'toilet_0090_32_x0_z0.binvox',  # too small
    'toilet_0234_32_x0_z0.binvox',  # too small
    'toilet_0082_32_x0_z0.binvox',  # just a circle
    'sofa_0455_32_x0_z0.binvox',
    'sofa_0507_32_x0_z0.binvox',
    'sofa_0553_32_x0_z0.binvox',
    'sofa_0255_32_x0_z0.binvox',
    'sofa_0015_32_x0_z0.binvox',
    'sofa_0037_32_x0_z0.binvox',
    'sofa_0287_32_x0_z0.binvox',
    'sofa_0376_32_x0_z0.binvox',
    'chair_0073_32_x0_z0.binvox',  # lots of gaps in chair back
    'chair_0619_32_x0_z0.binvox',  # strange front cross bar
    'chair_0722_32_x0_z0.binvox',  # lots of gaps in chair back
    'chair_0117_32_x0_z0.binvox',
    'chair_0982_32_x0_z0.binvox',
    'chair_0433_32_x0_z0.binvox',
    'chair_0053_32_x0_z0.binvox',
    'chair_0916_32_x0_z0.binvox',
    'chair_0668_32_x0_z0.binvox',
    'bed_0572_32_x0_z0.binvox',
    'bed_0159_32_x0_z0.binvox',
    'bed_0004_32_x0_z0.binvox',
    'bed_0367_32_x0_z0.binvox',
    'bed_0126_32_x0_z0.binvox',
    'bed_0581_32_x0_z0.binvox',
    'bed_0022_32_x0_z0.binvox',
    'bed_0341_32_x0_z0.binvox',
    'bed_0046_32_x0_z0.binvox',
    'bed_0042_32_x0_z0.binvox',
    'bed_0135_32_x0_z0.binvox',
    'bed_0263_32_x0_z0.binvox',
    'bed_0354_32_x0_z0.binvox',
    'bed_0412_32_x0_z0.binvox',
    'bed_0167_32_x0_z0.binvox',
    'bed_0429_32_x0_z0.binvox',
    'bed_0495_32_x0_z0.binvox',
    'bed_0102_32_x0_z0.binvox',
    'bed_0595_32_x0_z0.binvox',
    'bed_0436_32_x0_z0.binvox',
    'bed_0055_32_x0_z0.binvox',
    'bed_0454_32_x0_z0.binvox',
    'bed_0017_32_x0_z0.binvox',
    'bed_0307_32_x0_z0.binvox',
    'bed_0394_32_x0_z0.binvox',
    'bed_0410_32_x0_z0.binvox',
    'chair_0943_32_x0_z0.binvox',
]

KNOWN_INTERESTING_MODELNET10 = [
    'toilet_0375_32_x0_z0.binvox',
    'chair_0906_32_x0_z0.binvox',
    'chair_0667_32_x0_z0.binvox',
    'monitor_0562_32_x0_z0.binvox',
    'monitor_0531_32_x0_z0.binvox',
    'monitor_0340_32_x0_z0.binvox',
    'monitor_0507_32_x0_z0.binvox',
    'monitor_0163_32_x0_z0.binvox',
    'monitor_0456_32_x0_z0.binvox',
    'monitor_0496_32_x0_z0.binvox',
    'monitor_0405_32_x0_z0.binvox',
    'toilet_0268_32_x0_z0.binvox',
    'chair_0935_32_x0_z0.binvox',
    'toilet_0389_32_x0_z0.binvox',
    'monitor_0465_32_x0_z0.binvox',
    'monitor_0519_32_x0_z0.binvox',
    'chair_0641_32_x0_z0.binvox',
    'chair_0150_32_x0_z0.binvox',
    'sofa_0681_32_x0_z0.binvox',
    'chair_0747_32_x0_z0.binvox',
    'toilet_0276_32_x0_z0.binvox',
    'toilet_0434_32_x0_z0.binvox',
    'toilet_0277_32_x0_z0.binvox',
    'toilet_0191_32_x0_z0.binvox',
    'toilet_0014_32_x0_z0.binvox',
    'toilet_0263_32_x0_z0.binvox',
    'toilet_0280_32_x0_z0.binvox',
    'toilet_0341_32_x0_z0.binvox',
    'toilet_0354_32_x0_z0.binvox',
    'toilet_0016_32_x0_z0.binvox',
    'toilet_0357_32_x0_z0.binvox',
    'toilet_0229_32_x0_z0.binvox',
    'toilet_0072_32_x0_z0.binvox',
    'toilet_0031_32_x0_z0.binvox',
    'toilet_0217_32_x0_z0.binvox',
    'toilet_0383_32_x0_z0.binvox',
    'toilet_0358_32_x0_z0.binvox',
    'toilet_0312_32_x0_z0.binvox',
    'toilet_0266_32_x0_z0.binvox',
    'toilet_0382_32_x0_z0.binvox',
    'toilet_0064_32_x0_z0.binvox',
    'toilet_0051_32_x0_z0.binvox',
    'toilet_0177_32_x0_z0.binvox',
    'toilet_0364_32_x0_z0.binvox',
    'toilet_0197_32_x0_z0.binvox',
    'sofa_0333_32_x0_z0.binvox',
    'sofa_0573_32_x0_z0.binvox',
    'sofa_0183_32_x0_z0.binvox',
    'sofa_0486_32_x0_z0.binvox',
    'sofa_0339_32_x0_z0.binvox',
    'sofa_0757_32_x0_z0.binvox',
    'sofa_0103_32_x0_z0.binvox',
    'sofa_0160_32_x0_z0.binvox',
    'sofa_0502_32_x0_z0.binvox',
    'sofa_0368_32_x0_z0.binvox',
    'sofa_0266_32_x0_z0.binvox',
    'sofa_0051_32_x0_z0.binvox',
    'chair_0427_32_x0_z0.binvox',
    'chair_0637_32_x0_z0.binvox',
    'chair_0048_32_x0_z0.binvox',
    'chair_0913_32_x0_z0.binvox',
    'chair_0541_32_x0_z0.binvox',
    'chair_0407_32_x0_z0.binvox',
    'chair_0934_32_x0_z0.binvox',
    'chair_0896_32_x0_z0.binvox',
    'chair_0014_32_x0_z0.binvox',
    'chair_0011_32_x0_z0.binvox',
    'chair_0042_32_x0_z0.binvox',
    'chair_0595_32_x0_z0.binvox',
    'chair_0017_32_x0_z0.binvox',
    'bed_0558_32_x0_z0.binvox',
    'chair_0645_32_x0_z0.binvox',
    'chair_0370_32_x0_z0.binvox',
    'chair_0457_32_x0_z0.binvox',
    'toilet_0397_32_x0_z0.binvox',
    'toilet_0363_32_x0_z0.binvox',
    'toilet_0209_32_x0_z0.binvox',
    'toilet_0385_32_x0_z0.binvox'
]


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
                voxel_file = voxelize_file(model_file, ext='off', dest_dir=None, size=voxels_dim, verbose=True, timeout=120,
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
        
    def drop_bad_models(self, models=KNOWN_BAD_MODELNET10):
        for m in models:
            self.df = self.df[~self.df['binvox'].str.contains(m)]
        return
        
    def get_random_voxels(self, voxels_dim):
        random_sample = self.df.iloc[random.sample(list(self.df.index), 1)]
        #random_sample = self.df.sample(n=1).iloc[0]
        filename = random_sample['binvox'].iloc[0]
        voxels = self.get_voxels(random_sample['category'].iloc[0], random_sample['dataset'].iloc[0], filename)
        return filename, voxels

    def get_voxels_path(self, category, dataset, voxel_file):
        """
        """
        vox_path = os.path.join(MODELNET10_DIR, category, dataset, voxel_file)
        return vox_path
    
    def read_voxels(self, voxels_path, shape=None):
        # read in voxels
        vox = read_voxel_array(voxels_path)
        # convert from bool True/False to float 1/0 (tf likes floats)
        vox_data = vox.data.astype(np.float32)
        if shape:
            vox_data = np.reshape(vox_data, shape)
        return vox_data
        
    def get_voxels(self, category, dataset, voxel_file, verbose=False, shape=None):
        # construct path
        vox_path = self.get_voxels_path(category, dataset, voxel_file)
        if verbose:
            print('Voxel path: {}'.format(vox_path))
        vox_data = self.read_voxels(vox_path, shape=shape)
        return vox_data
    
    def filter_x_z(self, x, z):
        self.df = self.df[(self.df.x_rotations == x) & (self.df.z_rotations == z)]
        self.df = self.df.reset_index(drop=True)
        return
    
    def filter_categories(self, categories):
        df_filter = (self.df.category == categories[0])
        for cat in categories[1:]:
            df_filter = df_filter | (self.df.category == cat)
        if df_filter is not None:
            self.df = self.df[df_filter]
            self.df = self.df.reset_index(drop=True)
        return

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
            #if verbose:
            #    print('Creating dataset split for "{}"'.format(set_filter))
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
    
    def voxels_batchmaker_2(self, batch_size, voxels_dim, set_filter=None):
        """
        Test for mashup of two files
        Args:
            batch_size: int
            set_filter: str, "train" or "test" or None; returns specified dataset or both
            verbose: bool, extra debug prints
            pad: bool, pads final batch with all-zero examples or skips remainder
        """
        # generate a pair of files
        batch_1 = list()
        batch_2 = list()
        
        if set_filter in ['train', 'test']:
            df_slicer = self.df.dataset == set_filter
            #if verbose:
            #    print('Creating dataset split for "{}"'.format(set_filter))
        else:
            df_slicer = self.df == self.df

        for i, (index, row) in enumerate(self.df[df_slicer].iterrows()):
            vox_data = self.get_voxels(row['category'], row['dataset'], row['binvox'], shape=[voxels_dim, voxels_dim, voxels_dim, 1])
            if vox_data is None:
                continue
            if len(batch_1) < batch_size:
                batch_1.append(vox_data)
            elif len(batch_2) < batch_size:
                batch_2.append(vox_data)
 
            # yield batch if ready; else continue
            if (i+1) % (batch_size *2) == 0:
                yield np.asarray((batch_1, batch_2))
                batch_1 = list()
                batch_2 = list()

        return
        
    def __repr__(self):
        return '<ModelNet10()>'
