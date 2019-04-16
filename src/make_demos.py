"""
As the filename says, this script's goal is to make demos for ppt and report

We use the list in src/data/demos.py as input.
"""
# project imports
from utils import PROJECT_ROOT, read_json_data, flatten_dict, np_recon_loss, get_logger
from data.voxels import read_voxel_array, convert_voxels_to_stl, plot_voxels
from data.binvox_rw import write as write_binvox
from data import PROCESSED_DIR, MODELNET10_RECONS, MODELNET10_INDEX, MODELNET10_DIR, DOWNLOADED_STLS_DIR
from models.voxel_vaegan import VoxelVaegan
from data.modelnet10 import ModelNet10, KNOWN_NOT_INTERESTING_MODELNET10, KNOWN_INTERESTING_MODELNET10
from data.thingi10k import Thingi10k
from utils import read_json_data, DEMOS_DIR
from shutil import copyfile
from data.stl import plot_mesh, save_vectors_as_stl
from models import MODEL_DIR
from make_interesting_combinations import encode
from data import demos


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
VOXELS_DIM = 64
#VOXEL_THRESHOLD = 0.9
#ROTATE_CATEGORIES = ['sofa', 'monitor', 'dresser']
get_logger()


def combine_and_save_objects(obj1, obj2, dest_dir, vaegan, voxels_dim=32):
    """
    Returns:
        str, combo name of objects processed
    """
    basedir1 = os.path.dirname(obj1)
    basedir2 = os.path.dirname(obj2)
    name1 = os.path.splitext(os.path.basename(obj1))[0]
    name2 = os.path.splitext(os.path.basename(obj2))[0]
    path_voxels1 = os.path.join(basedir1, name1 + '.binvox')
    path_voxels2 = os.path.join(basedir2, name2 + '.binvox')
    
    # generate recons if we do not have already
    recon_and_save_object(obj1, dest_dir, vaegan, voxels_dim)
    recon_and_save_object(obj2, dest_dir, vaegan, voxels_dim)

    # prepare to save
    name_combo = '{}.{}'.format(name1, name2)
    dest_baseline_stl = os.path.join(dest_dir, '{}.baseline.stl'.format(name_combo))
    dest_baseline_voxels = os.path.join(dest_dir, '{}.baseline.binvox'.format(name_combo))
    dest_combo_stl = os.path.join(dest_dir, '{}.combo.stl'.format(name_combo))
    dest_combo_voxels = os.path.join(dest_dir, '{}.combo.stl'.format(name_combo))
    if (os.path.exists(dest_baseline_stl) and os.path.exists(dest_baseline_voxels) and
        os.path.exists(dest_combo_stl) and os.path.exists(dest_combo_voxels)):
        logging.info('recon already processed for {}; skipping'.format(name_combo))
        return name_combo
    
    # encode objs
    input1 = read_voxel_array(path_voxels1).data
    latent1 = encode(obj1, input1, vaegan, voxels_dim)
    input2 = read_voxel_array(path_voxels2).data
    latent2 = encode(obj2, input2, vaegan, voxels_dim)

    # combine and recon
    mid = latent1 + latent2    
    recon_voxels = vaegan.latent_recon(mid)
    recon_voxels = np.reshape(recon_voxels, [voxels_dim, voxels_dim, voxels_dim])

    # baseline
    baseline_voxels = input1 + input2
    baseline_stl = convert_voxels_to_stl(baseline_voxels)

    # final form
    recon_voxels_print = recon_voxels > .9
    recon_stl_print = convert_voxels_to_stl(recon_voxels_print)

    # save baseline
    save_vectors_as_stl(baseline_stl, dest_baseline_stl)
    logging.debug('dest_baseline_stl: {}'.format(dest_baseline_stl))
    np.save(dest_baseline_voxels, baseline_voxels)
    logging.debug('dest_baseline_voxels: {}'.format(dest_baseline_voxels))
    # save combo
    save_vectors_as_stl(recon_stl_print, dest_combo_stl)
    logging.debug('dest_combo_stl: {}'.format(dest_combo_stl))
    np.save(dest_combo_voxels, recon_voxels_print)
    logging.debug('dest_combo_voxels: {}'.format(dest_combo_voxels))

    return name_combo


def recon_and_save_object(obj, dest_dir, vaegan, voxels_dim=32):
    """
    Assumes that obj is an stl or binvox file and that accompanying stl
    or binvox file is sitting next to it in same dir
    
    Saves recon stl, recon binvox, orig stl, orig binvox in dest_dir
    
    If all of the above already exist, work is skipped
    
    Returns:
        str, name of object processed
    """
    basedir = os.path.dirname(obj)
    name = os.path.splitext(os.path.basename(obj))[0]
    # some inputs are .obj files; we handle that here
    stl_ext = '.stl'
    path_stl = os.path.join(basedir, name + '.stl')
    if not os.path.exists(path_stl):
        # warning: a little hacky -- check if this has _64 on the end for downloaded_stls
        tmp_stl = path_stl.replace('_{}.'.format(voxels_dim), '.')
        if os.path.exists(tmp_stl):
            path_stl = tmp_stl
        else:
            # warning: also a little hacky -- we assume name scheme for the modelnet shapes
            path_stl = path_stl.replace('_{}_x0_z0.stl'.format(voxels_dim), '.off')
            stl_ext = '.off'
            # and another hack to handle ShapeNet
            if 'ShapeNet' in path_stl and not os.path.exists(path_stl):
                path_stl = os.path.join(basedir, 'model_normalized.obj')
                stl_ext = '.obj'
    path_voxels = os.path.join(basedir, name + '.binvox')
    
    # build output paths; check if work is already done
    dest_voxels = os.path.join(dest_dir, '{}.recon.binvox'.format(name))
    dest_stl = os.path.join(dest_dir, '{}.recon.stl'.format(name))
    dest_orig_voxels = os.path.join(dest_dir, '{}.orig.binvox'.format(name))
    dest_orig_stl = os.path.join(dest_dir, '{}.orig{}'.format(name, stl_ext))
    if (os.path.exists(dest_voxels) and os.path.exists(dest_stl) and
        os.path.exists(dest_orig_voxels) and os.path.exists(dest_orig_stl)):
        logging.info('recon already processed for {}; skipping'.format(name))
        return

    # perform reconstruction
    obj_voxels = read_voxel_array(path_voxels).data
    print(path_voxels)
    obj_voxels = np.reshape(obj_voxels, (-1, voxels_dim, voxels_dim, voxels_dim, 1))
    recon_voxels = vaegan.reconstruct(obj_voxels)
    recon_voxels = np.reshape(recon_voxels, [voxels_dim, voxels_dim, voxels_dim])
    recon_print = recon_voxels > .9
    recon_stl = convert_voxels_to_stl(recon_print)

    # save recon vox
    logging.debug('dest_voxels: {}'.format(dest_voxels))
    np.save(dest_voxels, recon_print)  # recon_voxels or recon_print...?
    # save recon stl
    logging.debug('dest_stl: {}'.format(dest_stl))
    save_vectors_as_stl(recon_stl, dest_stl)
    # save orig vox
    copyfile(path_voxels, dest_orig_voxels)
    # save orig stl
    copyfile(path_stl, dest_orig_stl)

    return name


def main():

    ### model to use for reconstruction

    vae_modelnet10 = '/home/jcworkma/jack/3d-form/models/voxel_vae_modelnet10_64_200epochs'
    model_root = os.path.join(PROJECT_ROOT, vae_modelnet10)
    model_cfg = read_json_data(os.path.join(model_root, 'cfg.json'))
    model_ckpt = os.path.join(model_root, 'model_epoch-_end.ckpt')
    #model_ckpt = os.path.join(model_root, 'model_epoch-_end.ckpt')
    logging.info('model_cfg: {}'.format(model_cfg))
    logging.info('model_ckpt: {}'.format(model_ckpt))

    ### restore the model from ckpt

    vaegan = VoxelVaegan.initFromCfg(model_cfg)
    vaegan.restore(model_ckpt)
    print()
    print()

    ### Good Recons

    dest_dir = os.path.join(DEMOS_DIR, 'good_reconstructions')
    os.makedirs(dest_dir, exist_ok=True)
    good_recons = demos.good_recons_list(voxels_dim=VOXELS_DIM)
    target = len(good_recons)
    print(good_recons)
    for i, path_voxels in enumerate(good_recons):
        name = recon_and_save_object(path_voxels, dest_dir, vaegan, voxels_dim=VOXELS_DIM)
        logging.info('Completed #{}/{}: {}'.format(i, target, name))

    ### Bad Recons
    
    dest_dir = os.path.join(DEMOS_DIR, 'bad_reconstructions')
    os.makedirs(dest_dir, exist_ok=True)
    bad_recons = demos.bad_recons_list(voxels_dim=VOXELS_DIM)
    target = len(bad_recons)
    for path_voxels in bad_recons:
        logging.debug(path_voxels)
        name = recon_and_save_object(path_voxels, dest_dir, vaegan, voxels_dim=VOXELS_DIM)
        logging.info('Completed #{}/{}: {}'.format(i, target, name))

    ### Good Combos
        
    dest_dir = os.path.join(DEMOS_DIR, 'good_combos')
    os.makedirs(dest_dir, exist_ok=True)
    good_combos = demos.good_combos_list(voxels_dim=VOXELS_DIM)
    target = len(good_combos)
    for i, (path_voxels1, path_voxels2) in enumerate(good_combos):
        combod = combine_and_save_objects(path_voxels1, path_voxels2, dest_dir, vaegan, voxels_dim=VOXELS_DIM)
        logging.info('Completed #{}/{}: {}'.format(i, target, combod))

    ### Bad Combos
        
    dest_dir = os.path.join(DEMOS_DIR, 'bad_combos')
    os.makedirs(dest_dir, exist_ok=True)
    bad_combos = demos.bad_combos_list(voxels_dim=VOXELS_DIM)
    target = len(bad_combos)
    for i, (path_voxels1, path_voxels2) in enumerate(bad_combos):
        combod = combine_and_save_objects(path_voxels1, path_voxels2, dest_dir, vaegan, voxels_dim=VOXELS_DIM)
        logging.info('Completed #{}/{}: {}'.format(i, target, combod))

    return


if __name__ == '__main__':
    main()
