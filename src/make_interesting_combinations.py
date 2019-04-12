"""
As the filename says, this script's goal is to make interesting combinations

We use the list in modelnet10.py and the downloaded_stls as input. We then perform each and
every combination possible between all of those files and save them away for review.
"""

# project imports
from utils import PROJECT_ROOT, read_json_data, flatten_dict, np_recon_loss, get_logger
from data.voxels import read_voxel_array, convert_voxels_to_stl, plot_voxels
from data.binvox_rw import write as write_binvox
from data import PROCESSED_DIR, MODELNET10_RECONS, MODELNET10_INDEX, MODELNET10_DIR, DOWNLOADED_STLS_DIR
from models.voxel_vaegan import VoxelVaegan
from data.modelnet10 import ModelNet10, KNOWN_NOT_INTERESTING_MODELNET10, KNOWN_INTERESTING_MODELNET10
from data.thingi10k import Thingi10k
from utils import read_json_data
from shutil import copyfile
from data.stl import plot_mesh, save_vectors_as_stl
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


DEST_DIR = os.path.join(PROCESSED_DIR, 'INTERESTING_COMBOS')
DEST_DIR = '../../3d-form-output/combinations'

#tf.reset_default_graph()
VOXELS_DIM = 64
VOXEL_THRESHOLD = 0.9
ROTATE_CATEGORIES = ['sofa', 'monitor', 'dresser']
get_logger()


def encode(path, input1, vaegan, voxels_dim=VOXELS_DIM):
    if any(cat in path for cat in ROTATE_CATEGORIES):
        input1 = np.rot90(input1)
    input1 = np.reshape(input1, (-1, voxels_dim, voxels_dim, voxels_dim, 1))
    latent1 = vaegan.encode(input1)
    return latent1


def main():

    ### model to use for reconstruction

    model_root = '/home/jcworkma/jack/3d-form/models/voxel_vae_modelnet10_64_130epochs'
    model_cfg = read_json_data(os.path.join(model_root, 'cfg.json'))
    model_ckpt = os.path.join(model_root, 'model_epoch-129.ckpt')
    #model_ckpt = os.path.join(model_root, 'model_epoch-_end.ckpt')
    logging.info('model_cfg: {}'.format(model_cfg))
    logging.info('model_ckpt: {}'.format(model_ckpt))

    ### restore the model from ckpt

    vaegan = VoxelVaegan.initFromCfg(model_cfg)
    vaegan.restore(model_ckpt)

    ### loop through modelnet10, make recons

    modelnet = ModelNet10.initFromIndex(MODELNET10_INDEX)
    #modelnet.drop_bad_models()
    #modelnet.drop_bad_models(models=KNOWN_NOT_INTERESTING_MODELNET10)
    modelnet.keep_models(KNOWN_INTERESTING_MODELNET10)
    # filter to forward and straight orientation
    modelnet.filter_x_z(0, 0)
    modelnet_length = len(modelnet.df)
    modelnet.df['vox_path'] = MODELNET10_DIR + '/' + modelnet.df['category'] + '/' + modelnet.df['dataset'] + '/' + modelnet.df['binvox']
    logging.info('ModelNet categories to process: {}'.format(modelnet.df.category.unique()))
    logging.info('ModelNet objects to process = {}'.format(modelnet_length))
    downloaded_stls = [os.path.join(DOWNLOADED_STLS_DIR, x) for x in os.listdir(DOWNLOADED_STLS_DIR) if '{}.binvox'.format('' if VOXELS_DIM == 32 else '_{}'.format(VOXELS_DIM)) in x]
    logging.info('Downloaded STLs to process: {}'.format(len(downloaded_stls)))
    master_list = list(modelnet.df['vox_path']) + downloaded_stls
    logging.info('len(master_list) = {}'.format(len(master_list)))

    count = 0
    combos = defaultdict(list)
    for obj1 in master_list:
        for obj2 in master_list:
            if obj1 == obj2:
                # combining the same object won't do much for us :)
                continue
            if obj2 in combos[obj1] or obj1 in combos[obj2]:
                # we've done this one already!
                continue

            try:
                if not os.path.exists(obj1):
                    logging.warning('object does not exist: {}'.format(obj1))
                    continue
                if not os.path.exists(obj2):
                    logging.warning('object does not exist: {}'.format(obj2))
                    continue
                
                combos[obj1].append(obj2)
                
                # encode objs
                print(obj1)
                input1 = read_voxel_array(obj1).data
                latent1 = encode(obj1, input1, vaegan, VOXELS_DIM)
                print(obj2)
                input2 = read_voxel_array(obj2).data
                latent2 = encode(obj2, input2, vaegan, VOXELS_DIM)
                
                # combine and recon
                mid = latent1 + latent2    
                recon = vaegan.latent_recon(mid)
                recon = np.reshape(recon, [VOXELS_DIM, VOXELS_DIM, VOXELS_DIM])

                # baseline
                baseline = input1 + input2
                stl_baseline = convert_voxels_to_stl(baseline)
                
                # final form
                recon_threshold = recon > .9
                stl_obj1 = convert_voxels_to_stl(np.reshape(input1, (VOXELS_DIM, VOXELS_DIM, VOXELS_DIM)))
                stl_obj2 = convert_voxels_to_stl(np.reshape(input2, (VOXELS_DIM, VOXELS_DIM, VOXELS_DIM)))
                stl_recon = convert_voxels_to_stl(recon_threshold)
                
                # write to destination
                name1 = os.path.basename(os.path.splitext(obj1)[0])
                name2 = os.path.basename(os.path.splitext(obj2)[0])
                dest_obj = os.path.join(DEST_DIR, name1)
                os.makedirs(dest_obj, exist_ok=True)
                # combo stl
                dest_stl = os.path.join(dest_obj, '{}__{}.stl'.format(name1, name2))
                logging.debug('dest_stl: {}'.format(dest_stl))
                save_vectors_as_stl(stl_recon, dest_stl)
                # combo vox
                dest_vox = os.path.join(dest_obj, '{}__{}.binvox'.format(name1, name2))
                logging.debug('dest_vox: {}'.format(dest_vox))
                np.save(dest_vox, recon_threshold)
                # baseline, obj1, obj2
                dest_baseline_stl = dest_stl.replace('.stl', '.baseline.stl')
                save_vectors_as_stl(stl_baseline, dest_baseline_stl)
                logging.debug('dest_baseline_stl: {}'.format(dest_baseline_stl))
                dest_obj1_stl = os.path.join(dest_obj, os.path.basename(os.path.splitext(obj1)[0]) + '.stl')
                save_vectors_as_stl(stl_obj1, dest_obj1_stl)
                logging.debug('dest_obj1_stl: {}'.format(dest_obj1_stl))
                dest_obj2_stl = os.path.join(dest_obj, os.path.basename(os.path.splitext(obj2)[0]) + '.stl')
                save_vectors_as_stl(stl_obj2, dest_obj2_stl)
                logging.debug('dest_obj2_stl: {}'.format(dest_obj2_stl))
                
                count += 1
                logging.info('Completed #{}: {} <-> {}'.format(count, name1, name2))
            

            except Exception as exc:
                logging.exception(str(exc))


if __name__ == '__main__':
    main()
