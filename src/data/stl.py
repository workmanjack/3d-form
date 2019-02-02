from mpl_toolkits import mplot3d
from matplotlib import pyplot
from stl import mesh
import os


# override max count of triangles
import stl
stl.stl.MAX_COUNT = 2000000000


def plot_mesh(mesh_vectors):
    """
    TODO: save to file instead of .show()?
    """

    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure)

    # add the vectors to the plot
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh_vectors))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)

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

def write_stl_normal_vectors(vertex_vectors, output='write_stl_normal_vectors.stl'):
    """
    Args:
        vertex_vectors: np.array with shape 3x3
    """
    # apparently numpy-stl has an update_normals function
    model = mesh.Mesh(data={'normals': [], 'vectors': vertex_vectors}, calculate_normals=True)
    model.save(output)
    return
