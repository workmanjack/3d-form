# -*- coding: utf-8 -*-
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging
import click
import os


from data import DATA_DIR, THINGI10K_INDEX, THINGI10K_INDEX_100, THINGI10K_INDEX_10, THINGI10K_INDEX_1000
from data import MODELNET10_DIR, MODELNET10_INDEX, MODELNET10_TOILET_INDEX
from data.thingi10k import make_thingi10k_index
from data.modelnet10 import make_modelnet10_index


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # logger = logging.getLogger(__name__)
    # logger.info('making final data set from raw data')
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX_10, limit=10, get_json=True, get_img=True)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX_100, limit=100, get_img=False)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX_1000, limit=1000, get_img=False)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX, get_json=False, get_img=False)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX_10, limit=10)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX_100, limit=100)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX_1000, limit=1000)
    #make_thingi10k_index(DATA_DIR, THINGI10K_INDEX)
    make_modelnet10_index(MODELNET10_DIR, MODELNET10_TOILET_INDEX, categories=['toilet'])
    #make_modelnet10_index(MODELNET10_DIR, MODELNET10_INDEX)


if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
