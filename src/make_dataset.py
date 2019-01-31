# -*- coding: utf-8 -*-
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging
import click
import os


from data.thingi10k import make_thingi10k_index


DATA_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '../data')
THINGI10K_INDEX = os.path.join(DATA_DIR, 'processed/thingi10k_index.csv')


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
    make_thingi10k_index(DATA_DIR, THINGI10K_INDEX)


if __name__ == '__main__':
    # log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    # project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
