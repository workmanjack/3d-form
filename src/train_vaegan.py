# project imports
from utils import get_logger, read_json_data
from models.voxel_vaegan import VoxelVaegan
from data.thingi10k import Thingi10k
from data.modelnet10 import ModelNet10
from data.voxels import plot_voxels
from models import MODEL_DIR


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
    logging.info('Numpy random seed: {}'.format(np.random.get_state()[1][0]))
    
    ### Save Config
    
    cfg_path = os.path.join(MODEL_DIR, cfg.get('model').get('ckpt_dir'), 'cfg.json')
    with open(cfg_path, 'w') as outfile:
        json.dump(cfg, outfile)
    logging.info('Saved cfg: {}'.format(cfg_path))
    
    ### Get Dataset

    index_file = cfg.get('dataset').get('index')
    dataset_type = cfg.get('dataset').get('class')
    dataset_class = ModelNet10 if dataset_type == 'ModelNet10' else 'Thingi10k'
    logging.info('Dataset: {}'.format(dataset_class))
    pctile = cfg.get('dataset').get('pctile', None)
    logging.info('Using dataset index {} and pctile {}'.format(index_file, pctile))
    dataset = dataset_class.initFromIndex(index=index_file, pctile=pctile)
    logging.info('Shuffling dataset')
    dataset.shuffle()
    # apply filter
    tag = cfg.get('dataset').get('tag', None)
    if tag:
        logging.info('Filtering dataset by tag: {}'.format(tag))
        dataset.filter_by_tag(tag)
    filter_id = cfg.get('dataset').get('filter_id', None)
    if filter_id:
        logging.info('Filtering dataset by id: {}'.format(filter_id))
        thingi.filter_by_id(filter_id)
    n_input = len(dataset)
    logging.info('dataset n_input={}'.format(n_input))

    # grab batch size
    cfg_model = cfg.get('model')
    VOXELS_DIM = cfg_model.get('voxels_dim')
    BATCH_SIZE = cfg_model.get('batch_size')
    
    # split
    splits = cfg.get('dataset').get('splits', None)
    generator_cfg = cfg.get('generator')
    if dataset_class == Thingi10k and splits:
        # splits only supported by Thingi10k
        logging.info('Splitting Datasets')
        thingi_train, thingi_dev, thingi_test = dataset.split(splits['train'], splits['test'])
        logging.info('Train Length: {}'.format(len(thingi_train)))
        logging.info('Dev Length: {}'.format(len(thingi_dev)))
        logging.info('Test Length: {}'.format(len(thingi_test)))
        train_generator = lambda: thingi_train.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        dev_generator = lambda: thingi_dev.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        test_generator = lambda: thingi_test.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
    elif dataset_class == ModelNet10 and splits:
        logging.info('Splitting Datasets')
        train_generator = lambda: dataset.voxels_batchmaker(
            batch_size=BATCH_SIZE, set_filter='train', voxels_dim=VOXELS_DIM,
            verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        # using the "test" set as dev here so that we can track better across epochs
        # not technically 'correct' training proceder; perhaps switch out later
        dev_generator = lambda: dataset.voxels_batchmaker(
            batch_size=BATCH_SIZE, set_filter='test', voxels_dim=VOXELS_DIM,
            verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        test_generator = None
    else:
        train_generator = lambda: dataset.voxels_batchmaker(
            batch_size=BATCH_SIZE, voxels_dim=VOXELS_DIM, verbose=generator_cfg.get('verbose'), pad=generator_cfg.get('pad'))
        dev_generator = None
        test_generator = None
            
    ### Prepare for Training
    
    logging.info('Num input = {}'.format(n_input))
    logging.info('Num batches per epoch = {:.2f}'.format(n_input / BATCH_SIZE))
    
    example_input_id = cfg_model.get('example_input_id', None)
    if example_input_id:
        stl_example = thingi.get_stl_path(stl_id=cfg_model.get('example_input_id'))
        training_example = thingi.get_voxels(VOXELS_DIM,
                                             stl_file=stl_example)

        plot_voxels(training_example)
    
    ### Train
    
    tf.reset_default_graph()

    try:
        vaegan = VoxelVaegan.initFromCfg(cfg)
        
        if cfg_model.get('launch_tensorboard', False):
            logdir = 'current:{}'.format(vaegan.tb_dir)
            if cfg_model.get('tb_compare', False):
                for tb_compare in cfg_model.get('tb_compare'):
                    logdir += ',{}:{}'.format(tb_compare[0], tb_compare[1])
            tb_cmd = ['tensorboard', '--logdir', logdir]
            logging.info(tb_cmd)
            tb_proc = subprocess.Popen(tb_cmd, stdout=subprocess.PIPE)

        count = 0
        lr = cfg_model.get('learning_rate', None)
        if not isinstance(lr, list):
            lr = [(cfg_model.get('epochs'), lr)]
        elif isinstance(lr, tuple):
             lr = [lr]
                
        for epochs, rate in lr:

            # Use learning rate for set number of epochs
            # If set number is None, then continue for rest of total
            num_epochs = epochs if epochs is not None else int(cfg_model.get('epochs')) - count

            vaegan.train(train_generator,
                         dev_generator,
                         test_generator,
                         epochs=num_epochs,
                         input_repeats=cfg_model.get('input_repeats'),
                         display_step=cfg_model.get('display_step'),
                         save_step=cfg_model.get('save_step'),
                         dev_step=cfg_model.get('dev_step'),
                         enc_lr=cfg_model.get('enc_lr'),
                         dec_lr=cfg_model.get('dec_lr'),
                         dis_lr=cfg_model.get('dis_lr'))

            count += num_epochs
        
        vaegan._save_model_ckpt('_end')
        logging.info('Saved final checkpoint')
        
        try:
            ex.info['metrics'] = vaegan.metrics
            ex.info['model_dir'] = vaegan.ckpt_dir
        except:
            # fails if not running in an experiment... which is okay
            logging.info('Did not save experiment metrics')
            pass

    except Exception as exc:
        logging.exception('Failed to train vaegan')
        vaegan.close()
        raise(exc)
        
    ### Test Model
    
    if example_input_id:
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
