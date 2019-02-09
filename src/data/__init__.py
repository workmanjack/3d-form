import os
from utils import PROJECT_ROOT


DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, 'data'))
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
THINGI10K_STL_DIR = os.path.join(EXTERNAL_DIR, 'Thingi10k/raw_meshes')
THINGI10K_INDEX = os.path.join(PROCESSED_DIR, 'thingi10k_index.csv')
THINGI10K_INDEX_10 = os.path.join(PROCESSED_DIR, 'thingi10k_index_10.csv')
THINGI10K_INDEX_100 = os.path.join(PROCESSED_DIR, 'thingi10k_index_100.csv')
THINGI10K_INDEX_1000 = os.path.join(PROCESSED_DIR, 'thingi10k_index_1000.csv')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
