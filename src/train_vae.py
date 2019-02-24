# project imports
from models.voxel_vae import VoxelVae
from data.thingi10k import Thingi10k
from data.voxels import plot_voxels
from utils import get_logger


# python & package imports
import tensorflow as tf
import numpy as np
import os


# set seeds for reproducibility
np.random.seed(12)
tf.set_random_seed(12)


def main():

    logger = get_logger()
    logger.info('Starting train_vae main')
    
    ### Get Dataset

    thingi = Thingi10k.init10k(pctile=.9)
    # apply filter
    #thingi.filter_by_id(1351747)
    thingi.filter_by_tag('animal')
    #thingi.filter_to_just_one()
    #thingi = Thingi10k.init10()
    #thingi = Thingi10k.init10(pctile=.1)
    n_input = len(thingi)
    logger.info('Thingi10k n_input={}'.format(n_input))
    
    ### Prepare for Training

    VOXELS_DIM = 32
    BATCH_SIZE = 22
    logger.info('Num input = {}'.format(n_input))
    logger.info('Num batches per epoch = {:.2f}'.format(n_input / BATCH_SIZE))
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
        logger.exception('Failed to train vae')
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

    logger.info('Done train_vae.py main')

    return


if __name__ == '__main__':
    main()
