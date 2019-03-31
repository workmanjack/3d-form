# project imports
from utils import PROJECT_ROOT, read_json_data, flatten_dict, np_recon_loss, get_logger
from data.voxels import read_voxel_array, convert_voxels_to_stl, plot_voxels
from models.voxel_vaegan import VoxelVaegan
from data.modelnet10 import ModelNet10
from data.thingi10k import Thingi10k
from utils import read_json_data
from data.stl import plot_mesh
from data import PROCESSED_DIR
from models import MODEL_DIR


# python & package imports
from collections import defaultdict
import tensorflow as tf
import logging.config
import pandas as pd
import numpy as np
import json
import csv
import os


#tf.reset_default_graph()
VOXEL_DIM = 32
VOXEL_THRESHOLD = 0.9
get_logger()

### model to use for reconstruction

vae_modelnet10 = '/home/jcworkma/jack/3d-form/models/voxel_vae_modelnet10_200epochs_1'
model_root = os.path.join(PROJECT_ROOT, vae_modelnet10)
model_cfg = read_json_data(os.path.join(model_root, 'cfg.json'))
model_ckpt = os.path.join(model_root, 'model_epoch-124.ckpt')
#model_ckpt = os.path.join(model_root, 'model_epoch-_end.ckpt')
logging.info('model_cfg: {}'.format(model_cfg))
logging.info('model_ckpt: {}'.format(model_ckpt))

### restore the model from ckpt

vaegan = VoxelVaegan.initFromCfg(model_cfg)
vaegan.restore(model_ckpt)

### loop through modelnet10, make recons, save top 10 best reconstructions by loss

dataset_class = model_cfg.get('dataset').get('class')
logging.info('dataset_class: {}'.format(dataset_class))
dataset_index = model_cfg.get('dataset').get('index')
logging.info('dataset_index: {}'.format(dataset_index))
dataset_categories = model_cfg.get('dataset').get('categories', None)
logging.info('dataset_categories: {}'.format(dataset_categories))
modelnet = ModelNet10.initFromIndex(dataset_index)
if dataset_categories:
    modelnet.filter_categories(dataset_categories)
# filter to forward and straight orientation
modelnet.filter_x_z(0, 0)
modelnet_length = len(modelnet.df)
logging.info('ModelNet categories to process: {}'.format(modelnet.df.category.unique()))
logging.info('ModelNet objects to process = {}'.format(modelnet_length))

recon_rows = list()
for i, (index, row) in enumerate(modelnet.df.iterrows()):
    try:
        vox_path = modelnet.get_voxels_path(row['category'], row['dataset'], row['binvox'])
        vox_array = modelnet.read_voxels(vox_path)
        recon_input = np.reshape(vox_array, (-1, VOXEL_DIM, VOXEL_DIM, VOXEL_DIM, 1))
        recon_output = vaegan.reconstruct(recon_input)
        rmax = np.max(recon_output)
        rmin = np.min(recon_output)
        rmean = np.mean(recon_output)
        rloss = np_recon_loss(recon_input, recon_output)
        recon_rows.append(list(row) + [rmax, rmin, rmean, rloss])
        #recon_shaped = np.reshape(recon_output, [VOXEL_DIM, VOXEL_DIM, VOXEL_DIM])
        #recon_print = recon_shaped > VOXEL_THRESHOLD
        if i % 100 == 0:
            logging.info('Recons Processed = {} / {}'.format(i, modelnet_length))
    except Exception as exc:
        logging.error('recon {} failed because {}'.format(i, str(exc)))

output_csv = os.path.join(PROCESSED_DIR, 'recons.csv')
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(list(modelnet.df.columns) + ['max', 'min', 'mean', 'loss'])
    for row in recon_rows:
        csvwriter.writerow(row)
        
logging.info('recon info written to {}'.format(output_csv))
