# project imports
from utils import get_logger, read_json_data
from models.voxel_vae import VoxelVae
from data.thingi10k import Thingi10k
from data.voxels import plot_voxels


# python & package imports
from sacred.observers import FileStorageObserver
from sacred import Experiment
import tensorflow as tf
import numpy as np
import logging.config
import json
import sys
import os


ex = Experiment(name='voxel_vae')
ex.observers.append(FileStorageObserver.create('experiments'))
#ex.add_config('configs/config1.json')


# set seeds for reproducibility
np.random.seed(12)
tf.set_random_seed(12)


@ex.automain
def main(cfg):

    get_logger()
    logging.info('Starting train_vae main')
    
    ### Get Dataset

    index_file = cfg.get('dataset').get('index')
    pctile = cfg.get('dataset').get('pctile', None)
    logging.info('Using thingi10k with index {} and pctile {}'.format(index_file, pctile))
    thingi = Thingi10k.initFromIndex(index=index_file,
                                     pctile=pctile)
    # apply filter
    tag = cfg.get('dataset').get('tag', None)
    if tag:
        logging.info('Filtering thingi10k by tag: {}'.format(tag))
        thingi.filter_by_tag(tag)
    #thingi.filter_by_id(1351747)
    n_input = len(thingi)
    logging.info('Thingi10k n_input={}'.format(n_input))
    
    ### Prepare for Training

    cfg_voxel_vae = cfg.get('voxel_vae')
    VOXELS_DIM = cfg_voxel_vae.get('voxels_dim')
    BATCH_SIZE = cfg_voxel_vae.get('batch_size')
    
    logging.info('Num input = {}'.format(n_input))
    logging.info('Num batches per epoch = {:.2f}'.format(n_input / BATCH_SIZE))
    
    stl_example = thingi.get_stl_path(stl_id=cfg_voxel_vae.get('example_stl_id'))
    training_example = thingi.get_voxels(VOXELS_DIM,
                                         stl_file=stl_example)
    
    plot_voxels(training_example)
    
    ### Train
    
    tf.reset_default_graph()

    try:
        vae = VoxelVae(input_dim=VOXELS_DIM,
                       latent_dim=cfg_voxel_vae.get('latent_dim'),
                       learning_rate=cfg_voxel_vae.get('learning_rate'),
                       keep_prob=cfg_voxel_vae.get('keep_prob'),
                       kl_div_loss_weight=cfg_voxel_vae.get('kl_div_loss_weight'),
                       recon_loss_weight=cfg_voxel_vae.get('recon_loss_weight'),
                       verbose=cfg_voxel_vae.get('verbose'),
                       debug=cfg_voxel_vae.get('debug'))

        generator = lambda: thingi.voxels_batchmaker(batch_size=BATCH_SIZE,
                                                     voxels_dim=VOXELS_DIM,
                                                     verbose=cfg_voxel_vae.get('generator_verbose'))

        vae.train(generator,
                  epochs=cfg_voxel_vae.get('epochs'),
                  input_repeats=cfg_voxel_vae.get('input_repeats'),
                  display_step=cfg_voxel_vae.get('display_step'),
                  save_step=cfg_voxel_vae.get('save_step'))
        
        ex.info['metrics'] = vae.metrics
        ex.info['model_dir'] = vae.ckpt_dir

    except Exception as exc:
        logging.exception('Failed to train vae')
        vae.close()
        raise(exc)
        
    ### Test Model
    
    #vox_data = thingi.get_voxels(
    #    VOXELS_DIM,
    #    stl_file=stl_example,
    #    shape=[-1, VOXELS_DIM, VOXELS_DIM, VOXELS_DIM, 1])
    #recon = vae.reconstruct(vox_data)
    #recon = np.reshape(recon, [VOXELS_DIM, VOXELS_DIM, VOXELS_DIM])
    #recon = recon > cfg_voxel_vae.get('voxel_prob_threshold')
    #plot_voxels(recon)

    logging.info('Done train_vae.py main')

    return


#if __name__ == '__main__':
#s    main()
