# project imports
from stl.base import BaseMesh
from data import VOXELS_DIR
from stl import mesh


# python & package imports
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np
import subprocess
import os


# override max count of triangles
import stl
stl.stl.MAX_COUNT = 2000000000


# default dim size of binvox voxel objects
VOXEL_SIZE = 64


def plot_mesh(mesh_vectors, title=None):
    """
    TODO: save to file instead of .show()?
    """

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # add the vectors to the plot
    axes.add_collection3d(Poly3DCollection(mesh_vectors))

    # Auto scale to the mesh size
    scale = mesh_vectors.reshape([-1, 9]).flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    if title is not None:
        pyplot.title(title, pad=20)
    
    # Show the plot to the screen
    pyplot.show()


def read_mesh_vectors(stl_file):
    """
    Shortcut to the data we really care about inside of the stl
    """
    if not os.path.exists(stl_file):
        print('{} does not exist'.format(stl_file))
        return None
    if '.stl' not in stl_file:
        print('{} is not an stl file'.format(stl_file))
        return None
    model = mesh.Mesh.from_file(stl_file)
    return model.vectors


def save_vectors_as_stl(vectors, dest):
    """
    Saves the provided vectors as an stl file ready for 3d printing
    
    Args:
        vectors: np.array in shape (?, 3, 3)
        
    Returns: None
    """
    data = np.zeros(len(vectors), dtype=BaseMesh.dtype)
    new_stl = mesh.Mesh(data)
    new_stl.vectors = vectors
    new_stl.save(dest)
    return


def voxelize_stl(stl_path, dest_dir=VOXELS_DIR, check_if_exists=True, size=VOXEL_SIZE):
    """
    Converts an STL file into a voxel representation with binvox
    
    Args:
        stl_path: str, path to stl file to voxelize
        dest_dir: str, dir to write .binvox file to (default VOXELS_DIR)
        check_if_exists: bool, if True, will check and see if a .binvox file already exists
                               and return that rather than regenerate (default True)
        size: int, specify bounding box size of produced voxel object where arg N makes NxNxN
                   (default=VOXEL_SIZE)

    Returns:
        str, path to binvox file
    """
    binvox_output = stl_path.replace('.stl', '.binvox')
    binvox_dest = os.path.join(dest_dir, os.path.basename(binvox_output))
    exists = os.path.exists(binvox_dest)
    if check_if_exists and exists:
        print('Not Voxelizing: Binvox for {} already exists at {}'.format(stl_path, binvox_dest))
        return None
    elif exists:
        # overwrite binvox
        os.remove(binvox_dest)
    subprocess.run(['../src/data/binvox', '-cb', '-d', str(size), stl_path])
    # binvox will output the binvox file in the same dir as stl_path
    # here we move it to the desired dest
    os.rename(binvox_output, binvox_dest)
    return binvox_dest

