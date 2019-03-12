# project imports
from utils import get_logger, read_json_data
from models.voxel_vaegan import VoxelVaegan
from data.thingi10k import Thingi10k
from data.voxels import plot_voxels


# python & package imports
from sacred.observers import FileStorageObserver
from sacred import Experiment
import tensorflow as tf
import logging.config
import numpy as np
import subprocess
import json
import sys
import os


# set seeds for reproducibility
np.random.seed(12)
tf.set_random_seed(12)


# init sacred experiments
ex = Experiment(name='voxel_vaegan')
ex.observers.append(FileStorageObserver.create('experiments_vaegan'))


def train_vaegan(cfg):
    
    get_logger()
    logging.info('Starting train_vaegan main')
    
    ### Get Dataset

    index_file = cfg.get('dataset').get('index')
    pctile = cfg.get('dataset').get('pctile', None)
    logging.info('Using dataset index {} and pctile {}'.format(index_file, pctile))
    thingi = Thingi10k.initFromIndex(index=index_file, pctile=pctile)
    # apply filter
    tag = cfg.get('dataset').get('tag', None)
    if tag:
        logging.info('Filtering dataset by tag: {}'.format(tag))
        thingi.filter_by_tag(tag)
    filter_id = cfg.get('dataset').get('filter_id', None)
    if filter_id:
        logging.info('Filtering thingi10k by id: {}'.format(filter_id))
        thingi.filter_by_id(filter_id)
    n_input = len(thingi)
    logging.info('Thingi10k n_input={}'.format(n_input))
    
    # split
    splits = cfg.get('dataset').get('splits', None)
    generator_cfg = cfg.get('generator')
    if splits:
        logging.info('Splitting Datasets')
        thingi_train, thingi_dev, thingi_test = thingi.split(splits['train'], splits['test'])
        logging.info('Train Length: {}'.format(len(thingi_train)))
        logging.info('Dev Length: {}'.format(len(thingi_dev)))
        logging.info('Test Length: {}'.format(len(thingi_test)))
        train_generator = lambda: thingi_train.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        dev_generator = lambda: thingi_dev.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        test_generator = lambda: thingi_test.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
    else:
        train_generator = lambda: thingi.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        dev_generator = None
        test_generator = None
            
    ### Prepare for Training

    cfg_model = cfg.get('model')
    VOXELS_DIM = cfg_model.get('voxels_dim')
    BATCH_SIZE = cfg_model.get('batch_size')
    
    logging.info('Num input = {}'.format(n_input))
    logging.info('Num batches per epoch = {:.2f}'.format(n_input / BATCH_SIZE))
    
    example_stl_id = cfg_model.get('example_stl_id', None)
    if example_stl_id:
        stl_example = thingi.get_stl_path(stl_id=cfg_model.get('example_stl_id'))
        training_example = thingi.get_voxels(VOXELS_DIM,
                                             stl_file=stl_example)

        plot_voxels(training_example)
    
    ### Train
    
    tf.reset_default_graph()

    try:
        vaegan = VoxelVaegan.initFromCfg(cfg)
        
        if cfg_model.get('launch_tensorboard', False):
            tb_cmd = ['tensorboard', '--logdir', vaegan.tb_dir]
            logging.info(tb_cmd)
            tb_proc = subprocess.Popen(tb_cmd, stdout=subprocess.PIPE)

        vaegan.train(train_generator,
                     dev_generator,
                     test_generator,
                     epochs=cfg_model.get('epochs'),
                     input_repeats=cfg_model.get('input_repeats'),
                     display_step=cfg_model.get('display_step'),
                     save_step=cfg_model.get('save_step'),
                     dev_step=cfg_model.get('dev_step'))
        
        try:
            ex.info['metrics'] = vaegan.metrics
            ex.info['model_dir'] = vaegan.ckpt_dir
        except:
            # fails if not running in an experiment... which is okay
            print('Did not save experiment metrics')
            pass

    except Exception as exc:
        logging.exception('Failed to train vaegan')
        vaegan.close()
        raise(exc)
        
    ### Test Model
    
    if example_stl_id:
        vox_data = thingi.get_voxels(
            VOXELS_DIM,
            stl_file=stl_example,
            shape=[-1, VOXELS_DIM, VOXELS_DIM, VOXELS_DIM, 1])
        recon = vaegan.reconstruct(vox_data)
        recon = np.reshape(recon, [VOXELS_DIM, VOXELS_DIM, VOXELS_DIM])
        recon = recon > cfg_model.get('voxel_prob_threshold')
        plot_voxels(recon)

    logging.info('Done train_vaegan.py main')

    return


@ex.automain
def main(cfg):

    train_vaegan(cfg)


#if __name__ == '__main__':
#s    main()
