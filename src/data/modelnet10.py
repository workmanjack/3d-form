# project imports
from data.voxels import voxelize_file, read_voxel_array
from data import MODELNET10_DIR
from utils import api_json, dataframe_pctile_slice


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
    models_processed = list()
    for model_file in os.listdir(model_dir):
        if '.off' not in model_file:
            continue
        model_file = os.path.join(model_dir, model_file)
        model_info = [model_file]
        for x in range(4):
            for z in range(4):
                voxel_file = voxelize_file(model_file, ext='off', dest_dir=None, size=voxels_dim, verbose=True,
                                           num_rotx=x, num_rotz=z, binvox_suffix='_{}_x{}_z{}'.format(voxels_dim, x, z))
                model_info.append(voxel_file)
                models_processed.append(model_info)
    return models_processed


def make_modelnet10_index(root, output):
    """
    Args:
        root: str, root of ModelNet10 directory
    """
    modelnet_categories = os.listdir(root)
    modelnet_categories.remove('README.txt')
    models = list()
    for category in modelnet_categories:
        # test dir
        test_dir = os.path.join(root, category, 'test')
        models_processed = process_modelnet10_model_dir(test_dir)
        models_processed = [['test'] + m for m in models_processed]
        models.append(models_processed)
        # train dir
        train_dir = os.path.join(root, category, 'train')
        models_processed = process_modelnet10_model_dir(train_dir)
        models_processed = [['train'] + m for m in models_processed]
        models.append(models_processed)
    with open(output, 'wb', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in models:
            csvwriter.writerow(row)
    return
        

class ModelNet10(object):
    
    def __init__(self):
        return
    
    def __repr__(self):
        return '<ModelNet10()>'
