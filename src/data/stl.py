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

KNOWN_CANNOT_VOXELIZE = [
    '1228190', '65278', '65282', '44498', '43987', '43988', '65281', '65279', '43989',
    '315359', '1230687', '342378', '294010', '226639', '242579', '1005586', '1087141', 
    '98571', '151360', '174365', '1527408', '496800', '73020', '518090', '1087138', 
    '1423009', '518029', '1088138', '988104', '1087143', '147463', '688369', '1088214',
    '815484', '789801', '120477', '518087', '931889', '226669', '78481', '518079', 
    '111170', '518083', '471288', '518085', '1601763', '390065', '226633', '147729',
    '518082', '518037', '1527409', '147525', '41357', '226674', '372057', '518031', 
    '1088054', '1772312', '931902', '1087144', '1351747', '59197', '518038', '1088280',
    '498974', '1088051', '98546', '1368052', '372056', '461115', '372112', '86056',
    '518034', '147736', '439142', '165115', '356580', '1527416', '135771', '815485',
    '518088', '252683', '1706479', '199666', '1087134', '790253', '518089', '252784',
    '1088137', '1423085', '372055', '115423', '518035', '103354', '488051', '242237',
    '518095', '41246', '470465', '451870', '1088139', '518094', '45811', '522979',
    '1231079', '1074637', '1088218', '55280', '75147', '518084', '1088056', '91347',
    '1700791', '151376', '226685', '518032', '518091', '940414', '1088225', '252632',
    '1088281', '1005587', '461112', '252786', '688370', '794006', '1527417', '1717686',
    '471289', '45809', '1088217', '1088055', '518092', '1088053', '518036', '1088213',
    '527631', '1088142', '226684', '82803', '1231078', '1088279', '518033', '242236', 
    '1688588', '1706478', '562343', '1717685', '94192', '1088215', '376252', '518030',
    '65942', '252653', '1527410', '816587', '372058', '93842', '518081', '518086', 
    '1423014', '215386', '804299', '46012', '1088052', '93743', '41359', '1088141', 
    '261583', '112798', '815486', '135701', '226677', '39507', '1088216', '199665',
    '518080', '789800', '252640', '804302', '81363', '357854', '226679', '226683', 
    '471290', '1088140', '120628', '931901', '120628', '1422991', '518039', '97805',
    '133086'
]

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


def can_voxelize(stl_path):
    """
    Will tell you if this stl file can be voxelized or not by consulting a list constructed
    via past experiences
    """
    stl_id = os.path.splitext(os.path.basename(stl_path))[0]
    #print('STL_ID:', stl_id)
    can_voxel = not str(stl_id) in KNOWN_CANNOT_VOXELIZE
    return can_voxel


def voxelize_stl(stl_path, dest_dir=VOXELS_DIR, check_if_exists=True, size=VOXEL_SIZE, verbose=False, timeout=20):
    """
    Converts an STL file into a voxel representation with binvox
    
    Args:
        stl_path: str, path to stl file to voxelize
        dest_dir: str, dir to write .binvox file to (default VOXELS_DIR)
        check_if_exists: bool, if True, will check and see if a .binvox file already exists
                               and return that rather than regenerate (default True)
        size: int, specify bounding box size of produced voxel object where arg N makes NxNxN
                   (default=VOXEL_SIZE)
        verbose: bool, if true, prints out extra debug statements

    Returns:
        str, path to binvox file
    """
    # first make sure that this stl is voxelizeable
    if not can_voxelize(stl_path):
        return None
    binvox_output = stl_path.replace('.stl', '.binvox')
    binvox_dest = os.path.join(dest_dir, os.path.basename(binvox_output))
    exists = os.path.exists(binvox_dest)
    if check_if_exists and exists:
        if verbose:
            print('Not Voxelizing: Binvox for {} already exists at {}'.format(stl_path, binvox_dest))
        return binvox_dest
    elif exists:
        # overwrite binvox
        os.remove(binvox_dest)
    # check if parent directory exists
    binvox_dir = os.path.dirname(binvox_dest)
    if not os.path.exists(binvox_dir):
        os.makedirs(binvox_dir, exist_ok=True)
    # convert
    cmd = ['../src/data/binvox', '-cb', '-d', str(size), stl_path]
    if verbose:
        print('running -- {}'.format(' '.join(cmd)))
    try:
        subprocess.run(['../src/data/binvox', '-cb', '-d', str(size), stl_path], timeout=20)
    except subprocess.TimeoutExpired as texp:
        # marked as true because we want to track these and add them to the KNOWN_CANNOT_VOXELIZE list
        if True or verbose:
            print('conversion timed out for {}'.format(stl_path))
    # binvox will output the binvox file in the same dir as stl_path
    # check to make sure it worked
    if not os.path.exists(binvox_output):
        if verbose:
            print('binvox failed to convert {}'.format(stl_path))
        binvox_dest = None
    else:
        # here we move it to the desired dest
        os.rename(binvox_output, binvox_dest)
    return binvox_dest

