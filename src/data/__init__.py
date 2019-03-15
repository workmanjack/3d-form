import os
from utils import PROJECT_ROOT


DATA_DIR = os.path.join(os.path.join(PROJECT_ROOT, 'data'))
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
VOXELS_DIR = os.path.join(PROCESSED_DIR, 'Thingi10k/voxels')
EXTERNAL_DIR = os.path.join(DATA_DIR, 'external')
THINGI10K_STL_DIR = os.path.join(EXTERNAL_DIR, 'Thingi10k/raw_meshes')
THINGI10K_INDEX = os.path.join(PROCESSED_DIR, 'thingi10k_index.csv')
THINGI10K_INDEX_10 = os.path.join(PROCESSED_DIR, 'thingi10k_index_10.csv')
THINGI10K_INDEX_100 = os.path.join(PROCESSED_DIR, 'thingi10k_index_100.csv')
THINGI10K_INDEX_1000 = os.path.join(PROCESSED_DIR, 'thingi10k_index_1000.csv')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
MODELNET10_DIR = os.path.join(EXTERNAL_DIR, 'ModelNet10')
MODELNET10_INDEX = os.path.join(PROCESSED_DIR, 'modelnet10_index.csv')
MODELNET10_TOILET_INDEX = os.path.join(PROCESSED_DIR, 'modelnet10_toilet_index.csv')
MODELNET10_SOFA_INDEX = os.path.join(PROCESSED_DIR, 'modelnet10_sofa_index.csv')
MODELNET10_BATHTUB_INDEX = os.path.join(PROCESSED_DIR, 'modelnet10_bathtub_index.csv')
MODELNET10_SOFA_TOILET_INDEX = os.path.join(PROCESSED_DIR, 'modelnet10_sofa-toilet_index.csv')
BINVOX = os.path.join(PROJECT_ROOT, 'src', 'data', 'binvox')
