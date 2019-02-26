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
import os


ex = Experiment()
ex.observers.append(FileStorageObserver.create('experiments'))
ex.add_config('configs/config1.json')


# set seeds for reproducibility
np.random.seed(12)
tf.set_random_seed(12)


@ex.automain
def main(cfg):

    get_logger()
    logging.info('Starting train_vae main')
    
    ### Get Dataset

    thingi = Thingi10k.initFromIndex(index=cfg.get('dataset').get('index'),
                                     pctile=cfg.get('dataset').get('pctile', None))
    # apply filter
    #thingi.filter_by_id(1351747)
    tag = cfg.get('dataset').get('tag', None)
    if tag:
        thingi.filter_by_tag(tag)
    n_input = len(thingi)
    logging.info('Thingi10k n_input={}'.format(n_input))
    
    return
    ### Prepare for Training

    VOXELS_DIM = 32
    BATCH_SIZE = 22
    logging.info('Num input = {}'.format(n_input))
    logging.info('Num batches per epoch = {:.2f}'.format(n_input / BATCH_SIZE))
    training_example = thingi.get_voxels(VOXELS_DIM, stl_file=thingi.get_stl_path(stl_id=126660))
    plot_voxels(training_example)
    
    ### Train
    
    tf.reset_default_graph()

    try:
        vae = VoxelVae(input_dim=VOXELS_DIM,
                                     latent_dim=100,
                                     learning_rate=0.0001,
                                     keep_prob=1.0,
                                     kl_div_loss_weight=1,
                                     recon_loss_weight=1e4,
                                     verbose=True,
                                     debug=True)

        generator = lambda: thingi.voxels_batchmaker(batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=False)

        vae.train(generator, epochs=50, input_repeats=1, display_step=1, save_step=10)
    except Exception as exc:
        logging.exception('Failed to train vae')
        vae.close()
        raise(exc)
        
    ### Test Model
    
    vox_data = thingi.get_voxels(
        VOXELS_DIM,
        stl_file=thingi.get_stl_path(stl_id=126660),
        shape=[-1, VOXELS_DIM, VOXELS_DIM, VOXELS_DIM, 1])
    recon = vae.reconstruct(vox_data)
    recon = np.reshape(recon, [VOXELS_DIM, VOXELS_DIM, VOXELS_DIM])
    recon = recon > 0.065
    plot_voxels(recon)

    logging.info('Done train_vae.py main')

    return


if __name__ == '__main__':
    main()
