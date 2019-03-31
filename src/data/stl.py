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


def plot_mesh(mesh_vectors, title=None, facecolor='red', edgecolor='black'):
    """
    TODO: save to file instead of .show()?
    """

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # add the vectors to the plot
    # 3D Collection docs: https://matplotlib.org/api/_as_gen/mpl_toolkits.mplot3d.art3d.Poly3DCollection.html
    collection3d = Poly3DCollection(mesh_vectors)
    collection3d.set_edgecolor(edgecolor)
    collection3d.set_facecolor(facecolor)
    axes.add_collection3d(collection3d)

    # Auto scale to the mesh size
    scale = mesh_vectors.reshape([-1, 9]).flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

    if title is not None:
        pyplot.title(title, pad=20)
    
    # Show the plot to the screen
    #pyplot.show()
    return pyplot


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
