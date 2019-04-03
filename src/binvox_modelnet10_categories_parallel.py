"""
This script was written to handle modelnet10 categories in parallel

A separate script was required because python can only support with subprocess (thanks GIL!),
and we needed to add command line args to the process dir call
"""
import subprocess
from data.voxels import voxelize_file
from data.modelnet10 import process_modelnet10_model_dir


if __name__ == '__main__':
    # get arguments
    # TODO: upgrade to argparse or something similar
    import sys
    args = sys.argv
    model_dir = args[1]
    voxels_dim = args[2]
    process_modelnet10_model_dir(model_dir, voxels_dim)
    