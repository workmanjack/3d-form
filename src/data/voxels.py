# project imports
from data.binvox_rw import read_as_3d_array


# python & package imports
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot
from skimage import measure
import numpy as np
import subprocess
import os


def read_voxel_array(vox_file):
    with open(vox_file, 'rb') as f:
        vox = read_as_3d_array(f)
    return vox


def plot_voxels(vox_data, title=None, figsize=(8,6), color='red'):
    """
    Uses matplotlib to create a 3D plot of the provided voxels
    
    Args:
        vox_data: np.array, cubic array with 1/True for voxel and 0/False for empty space
        title: (optional) str, title of plot
        figsize: (optional) tuple of 2 ints, dimension of figure
        color: (optional), color of voxels
        
    Returns:
        matplotlib pyplot object
    """    
    fig = pyplot.figure(figsize=figsize)
    
    ax = fig.gca(projection='3d')
    ax.voxels(vox_data, facecolors=color, edgecolor='k')

    if title is not None:
        pyplot.title(title, pad=20)
    
    #pyplot.show()
    return pyplot


def convert_voxels_to_stl(vox_data):

    # Use marching cubes to obtain the surface mesh
    verts, faces, normals, values = measure.marching_cubes_lewiner(vox_data, 0)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = pyplot.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # return mesh vectors
    return verts[faces]

    ### code for generating plot of these vectors
    ## Fancy indexing: `verts[faces]` to generate a collection of triangles
    #mesh = Poly3DCollection(verts[faces])
    #mesh.set_edgecolor('k')
    #ax.add_collection3d(mesh)

    #ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    #ax.set_ylabel("y-axis: b = 10")
    #ax.set_zlabel("z-axis: c = 16")

    #ax.set_xlim(0, vox.dims[0])
    #ax.set_ylim(0, vox.dims[1])
    #ax.set_zlim(0, vox.dims[2])

    #pyplot.tight_layout()
    #pyplot.show()
