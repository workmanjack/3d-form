import os


DATA_DIR = os.path.join(os.path.realpath(os.path.dirname(__file__)), '../../data')
THINGI10K_INDEX = os.path.join(DATA_DIR, 'processed/thingi10k_index.csv')
THINGI10K_INDEX_100 = os.path.join(DATA_DIR, 'processed/thingi10k_index_100.csv')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
