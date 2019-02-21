# project imports
from data.stl import read_mesh_vectors, voxelize_stl
from data.voxels import read_voxel_array
from data import RAW_DIR, VOXELS_DIR, THINGI10K_STL_DIR, THINGI10K_INDEX, THINGI10K_INDEX_10, THINGI10K_INDEX_100, THINGI10K_INDEX_1000
from utils import api_json, dataframe_pctile_slice


# python packages
import urllib.request
import pandas as pd
import numpy as np
import traceback
import shutil
import json
import time
import csv
import os


def download_thingi01k_api_data(obj_id):
    """
    Returns api data of specified thingi10k object

    Args:
        obj_id: int or str
    """
    url = 'https://ten-thousand-models.appspot.com/api/v1/model/{}'.format(obj_id)
    return api_json(url)


def download_thingi10k_image(obj_id, dest):
    """
    Returns an image file for the specified thingi10k object

    https://stackoverflow.com/questions/3042757/downloading-a-picture-via-urllib-and-python
    https://docs.python.org/3.0/library/urllib.request.html#urllib.request.urlretrieve

    Args:
        obj_id: int or str
        dest: str, path to save image file to

    Returns:
        str, path at which image file was saved
    """
    url = 'https://storage.googleapis.com/thingi10k/renderings/{}.png'.format(obj_id)
    print('img_query: {}'.format(url))
    filename, headers = urllib.request.urlretrieve(url)
    shutil.move(filename, dest)
    return dest


def make_thingi10k_index(data_dir, index_path, limit=None, get_json=True, get_img=True):
    """
    Constructs a csv index of thingi10k objects
    """
    print('Making thingi10k index from data in {} and saving to {}'.format(data_dir, index_path))

    rows = list()
    header = ['file', 'name', 'num_vertices']

    mesh_dir = os.path.join(data_dir, 'external/Thingi10k/raw_meshes')
    raw_dir = os.path.join(data_dir, 'raw')
    proc_dir = os.path.join(data_dir, 'processed')

    fields = list()
    data = list()

    count = 1

    start = time.time()

    files = os.listdir(mesh_dir)
    files = files if not limit else files[:limit]

    for path in files:

        try:

            # quick check that we are reading something that we care about
            if not path.endswith('.stl'):
                print('not an stl file: {}'.format(path))
                continue

            file_data = dict()
            file_data['stl_file'] = path
            print('{}: {}'.format(count, path))

            # get file ids
            obj_id = os.path.splitext(path)[0]
            
            # get normalization data
            vectors = read_mesh_vectors(os.path.join(mesh_dir, path))
            file_data['xyz_min'] = np.amin(vectors).item()
            file_data['xyz_max'] = np.amax(vectors).item()

            if get_img:
                # gather object images
                img_name = '{}.png'.format(obj_id)
                dest = os.path.join(raw_dir, img_name)
                if os.path.exists(dest):
                    print('{} already exists'.format(dest))
                else:
                    dest = download_thingi10k_image(obj_id, dest)
                file_data['img_file'] = img_name

            if get_json:
                # get api data
                api_data = download_thingi01k_api_data(obj_id)
                file_data = {**file_data, **api_data}
                json_name = '{}.json'.format(obj_id)
                file_data['json_file'] = json_name
                # write json
                dest = os.path.join(raw_dir, json_name)
                with open(dest, 'w') as outfile:
                    json.dump(file_data, outfile)

            # save index row
            fields = list(set(fields).union(set(file_data.keys())))
            data.append(file_data)
            count += 1

        except Exception:
            print('Problem!')
            print(traceback.format_exc())

    # write index
    # index_path = '_output/thingi10k.csv'
    with open(index_path, 'w', newline='') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fields)

        writer.writeheader()
        for i, obj in enumerate(data):
            # https://stackoverflow.com/questions/1285911/how-do-i-check-that-multiple-keys-are-in-a-dict-in-a-single-passs
            # make sure that this obj has all the right fields
            if set(fields) <= set(obj):
                for f in fields:
                    if not obj.get(f, None):
                        obj[f] = None

            writer.writerow(obj)

    print('{} objects processed'.format(count))
    print('Index written to {}'.format(index_path))


class Thingi10k(object):
    
    def __init__(self, df, index, pctile, stl_dir=THINGI10K_STL_DIR):
        self.index = index
        self.df = df
        self.pctile = pctile
        self.stl_dir = stl_dir
        return 
    
    @classmethod
    def initFromIndex(cls, index, pctile=None):
        """
        Create a pandas df of the provided index file

        Args:
            index: str, path to thingi10k index file
            pctile: float, limit selection to at or below this pctile of num_vertices (0 to 1)

        Returns:
            pd.DataFrame
        """
        df = pd.read_csv(index)
        if pctile:
            df = dataframe_pctile_slice(df, 'num_vertices', pctile)
        return cls(df, index, pctile)
            
    @classmethod
    def init1000(cls, pctile=None):
        return cls.initFromIndex(THINGI10K_INDEX_1000, pctile)

    @classmethod
    def init100(cls, pctile=None):
        return cls.initFromIndex(THINGI10K_INDEX_100, pctile)
    
    @classmethod
    def init10(cls, pctile=None):
        return cls.initFromIndex(THINGI10K_INDEX_10, pctile)

    @classmethod
    def init10k(cls, pctile=None):
        return cls.initFromIndex(THINGI10K_INDEX, pctile)

    def filter_to_just_one(self):
        self.df = self.df[:1]
        return
    
    def filter_by_tag(self, tag):
        """
        Filter dataframe rows down to those that contain "tag"
        
        Because the tags in the index look like ['tag1', 'tag2'] we wrap the provided
        tag in quotes to ensure an exact match.
        """
        # drop na first because we know those don't contain tag
        # and because they will break str.contains
        self.df = self.df.dropna(subset=['tags'])
        # filter
        self.df = self.df[self.df.tags.str.contains("'{}'".format(tag))]
        return
    
    def max_length(self):
        max_length = self.df['num_faces'].max() * 9
        return int(max_length)
    
    def num_triangles(self):
        return self.df.num_faces.sum()
    
    def _prep_normalization(self):
        xyz_min = self.df.xyz_min.min()
        xyz_max = self.df.xyz_max.max()
        return xyz_min, xyz_max
       
    def _normalize_vectors(self, vectors, xyz_min, xyz_max):
        vectors = (vectors - xyz_min) / (xyz_max - xyz_min) 
        return vectors

    def _triangulize_vectors(self, vectors):
        """
        Reshape vectors to Nx9 shape
        
        Elements should be [x1, y1, z1, x2, y2, z2, x3, y3, z3]
        """
        vectors = np.reshape(vectors, [-1, 9])
        return vectors
    
    def _flatten_vectors(self, vectors):
        """
        Flatten vectors to single dim
        """
        vectors = np.reshape(vectors, [-1])
        return vectors
        
    def _reform_vectors(self, vectors):
        """
        Reform vectors to traditional Nx3x3 shape
        """
        vectors = np.reshape(vectors, [-1, 3, 3])
        return vectors
        
    def _pad_vectors(self, vectors, pad_length):
        """
        Pads vectors to desired length with zeros

        Note: length is calculated from flattened array so must make pad_length %% 9 = 0

        Args:
            vectors: np.array Nx3x3 or N
            pad_length: int, each item will be padded with zeros until specified length is reached

        Returns:
            padded vectors of len=pad_length
        """
        if pad_length % 9 != 0:
            print('error! pad_length {} is not divisible by 9! will not pad!'.format(pad_length))
            return vectors
        vectors = self._flatten_vectors(vectors)
        num_zeros = pad_length - len(vectors)
        if num_zeros < 0:
            vectors = vectors[:pad_length]
        else:
            vectors = np.concatenate((vectors, np.zeros(num_zeros)), axis=None)
        vectors = self._reform_vectors(vectors)
        return vectors

    def batchmaker(self, batch_size, normalize=False, triangles=False, flat=False, pad_length=None, filenames=False):
        """
        Batch Generator for the Thingi10k dataset

        Args:
            batch_size: int, size of batches to produce
            flat: bool, if False, batches will be NxTx3x3 where N is batch_size and T is num triangles
                        if True, batches will be NxT*9
            pad_length: int, if provided, each item will be padded with zeros until specified length is reached
                             if an item is longer than pad_length, then it will be truncated
            filenames: bool, include filename in return or not

        Returns:
            ndarray of length batch_size and format (<stl_file>, <vectors>) if filenames else <vectors>
        """
        batch = list()
        xyz_min, xyz_max = self._prep_normalization()
        for i, stl_file in enumerate(self.df.stl_file):
            # read in stl file, read in vectors, apply ops as instructed
            stl_path = os.path.join(self.stl_dir, stl_file)
            vectors = read_mesh_vectors(stl_path)
            if pad_length:
                vectors = self._pad_vectors(vectors, pad_length)
            if flat:
                vectors = self._flatten_vectors(vectors)
            if triangles:
                vectors = self._triangulize_vectors(vectors)
            if normalize:
                vectors = self._normalize_vectors(vectors, xyz_min, xyz_max)
            if filenames:
                batch.append((stl_path, vectors))
            else:
                batch.append(vectors)
            # yield batch if ready; else continue
            if (i+1) % batch_size == 0:
                yield np.asarray(batch)
                batch = list()
        return
    
    def voxels_batchmaker(self, batch_size, voxels_dim, verbose=False):
        batch = list()
        for i, stl_file in enumerate(self.df.stl_file):
            # read in stl file, read in vectors, apply ops as instructed
            stl_path = os.path.join(self.stl_dir, stl_file)
            vox_path = voxelize_stl(stl_path, dest_dir=os.path.join(VOXELS_DIR, str(voxels_dim)), size=voxels_dim, verbose=verbose)
            if vox_path is None:
                # sometimes voxelization can fail, so we skip and move on to the next one
                continue
            vox = read_voxel_array(vox_path)
            # convert from bool True/False to float 1/0 (tf likes floats)
            vox_data = vox.data.astype(np.float32)
            # each element has 1 "channel" aka data point (if RGB color, it would be 3)
            batch.append(np.reshape(vox_data, [voxels_dim, voxels_dim, voxels_dim, 1])) 
            # yield batch if ready; else continue
            if (i+1) % batch_size == 0:
                yield np.asarray(batch)
                batch = list()
        return
        
    def triangle_sequencer(self, seq_length=1, normalize=True, pad_sequences=True):
        """
        Sequence Generator by Triangle for the Thingi10k dataset

        Args:
            seq_length: int, number of triangles to return at one time
            normalize: bool, normalize vertex coordinate values or not
            pad_sequences: bool, add zeros to complete a sequence if num triangles remaining
                                 is not enough to complete a sequence

        Returns:
            nd.array of shape (1, 3) where elements are [x, y, z]
        """
        xyz_min, xyz_max = self._prep_normalization()
        for i, stl_file in enumerate(self.df.stl_file):
            # read in stl file, read in vectors, apply ops as instructed
            stl_path = os.path.join(self.stl_dir, stl_file)
            vectors = read_mesh_vectors(stl_path)
            if normalize:
                vectors = self._normalize_vectors(vectors, xyz_min, xyz_max)
            triangles = self._triangulize_vectors(vectors)

            seq_count = 0
            to_yield = list()
            for tri in triangles:
                if (seq_length - seq_count) == 0:
                    yield np.asarray(to_yield)
                    seq_count = 0
                    to_yield = list()
                else:
                    to_yield.append(tri)
                    seq_count += 1
            if pad_sequences:
                while (seq_length - seq_count) > 0:
                    to_yield.append([0]*9)
                    seq_count += 1
                yield np.asarray(to_yield)
        return

    def triangle_batchmaker(self, batch_size, seq_length, pad_batches=True):
        """
        """
        batch = list()
        batch_count = 0
        for seq in self.triangle_sequencer(seq_length):
            if (batch_size - batch_count) == 0:
                yield np.asarray(batch)
                batch_count = 0
                batch = list()
            else:
                batch.append(seq)
                batch_count += 1
        if pad_batches:
            while (batch_size - batch_count) > 0:
                batch.append([[0]*9]*seq_length)
                batch_count += 1
            yield np.asarray(batch)
        if batch_count > 0 and batch_count != batch_size:
            print('leftovers: batch_count * seq_length = {} triangles left behind'.format(batch_count, seq_length))
        return
    
    def vertex_batchmaker(self, normalize=True):
        """
        Batch Generator by Vertex for the Thingi10k dataset

        Args:
            normalize: bool, normalize vertex coordinate values or not

        Returns:
            nd.array of shape (1, 3) where elements are [x, y, z]
        """
        batch = list()
        xyz_min, xyz_max = self._prep_normalization()
        for i, stl_file in enumerate(self.df.stl_file):
            # read in stl file, read in vectors, apply ops as instructed
            stl_path = os.path.join(self.stl_dir, stl_file)
            vectors = read_mesh_vectors(stl_path)
            if normalize:
                vectors = self._normalize_vectors(vectors, xyz_min, xyz_max)
            vertices = self._flatten_vectors(vectors)
            for vtx in vertices:
                yield vtx
        return

    def get_stl_title(self, stl_id):
        """
        """
        return self.df.loc[self.df.stl_file == '{}.stl'.format(stl_id), 'title'].iloc[0]
    
    def vectors(self, n=None, stl_id=None):
        """
        Retrieves the vectors of the specified stl file

        Note: requires n or stl_id (not both)

        Args:
            n: int, row index of stl object to return
            stl_id: int, stl id number from dataset of object to return
            
        Returns:
            vectors of requested object
        """
        if n and stl_id:
            raise ValueException('Thingi10k.vectors requires n or stl_id (not both)')
        stl_path = None
        if n:
            stl_path = self[n].stl_file
        else:
            df_stl = df[df.stl_file.str.contains(str(stl_id))]
            if len(df_stl) > 1:
                print('warning! more than one stl file matches stl_id={}; returning first match'.format(stl_id))
            stl_path = df_stl.iloc[0].stl_file
        stl_path = os.path.join(THINGI10K_STL_DIR, stl_path)
        vectors = read_mesh_vectors(stl_path)
        return vectors
    
    def get_stl_path(self, n):
        return os.path.join(self.stl_dir, self[n].stl_file)
    
    def __getitem__(self, n):
        return self.df.loc[n]
    
    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return '<Thingi10k(index={}, n={}, pctile={})'.format(self.index, len(self), self.pctile)
    