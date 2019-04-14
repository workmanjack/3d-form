"""
Quick dirty script to convert saved output reconstruction files into pngs
"""
import os
import sys
import numpy as np
from data.voxels import plot_voxels, read_voxel_array
from data.stl import plot_mesh, read_mesh_vectors


# first arg following script name
root_dir = sys.argv[1]

FIGSIZE = (10, 8)


def convert(fext, fpath, readfunc, plotfunc):
    count = 0
    if fpath.endswith(fext):
        dest = fpath.replace(fext, '.png')
        if not os.path.exists(dest):
            data = readfunc()
            fig = plotfunc(data, figsize=FIGSIZE)
            fig.savefig(dest)
            fig.close('all')
            count = 1
    return count
    

def main():
    count = 0
    all_files = os.listdir(root_dir)
    file_count = 0
    # how many are we doing?
    for i, f in enumerate(all_files):
        root2 = os.path.join(root_dir, f)
        file_count += len(os.listdir(root2))
    # now do the conversions
    for i, f in enumerate(all_files):
        root2 = os.path.join(root_dir, f)
        # go two levels deep
        for i, f in enumerate(os.listdir(root2)):
            fpath = os.path.join(root2, f)
            count += convert('.binvox', fpath, lambda: read_voxel_array(fpath, array=True), plot_voxels)
            count += convert('.npy', fpath, lambda: np.load(fpath), plot_voxels)
            count += convert('.stl', fpath, lambda: read_mesh_vectors(fpath), plot_mesh)
            print('{} / {} completed with {} conversions'.format(i, file_count, count)) 


if __name__ == '__main__':
    main()
