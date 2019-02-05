# project imports
from data.stl import read_mesh_vectors
from data import RAW_DIR, THINGI10K_STL_DIR, THINGI10K_INDEX, THINGI10K_INDEX_10, THINGI10K_INDEX_100, THINGI10K_INDEX_1000
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
            mins = np.amin(vectors[0], axis=0)
            maxs = np.amax(vectors[0], axis=0)
            file_data['x_min'] = mins[0].item()
            file_data['x_max'] = maxs[0].item()
            file_data['y_min'] = mins[1].item()
            file_data['y_max'] = maxs[1].item()
            file_data['z_min'] = mins[2].item()
            file_data['z_max'] = maxs[2].item()

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
    
    def __init__(self, df, index, pctile):
        self.index = index
        self.df = df
        self.pctile = pctile
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

    def max_length(self):
        max_length = self.df['num_faces'].max() * 9
        return int(max_length)
    
    def _prep_normalization(self):
        mins = np.asarray(self.df.x_min.min(), self.df.y_min.min(), self.df.z_min.min())
        maxs = np.asarray(self.df.x_max.max(), self.df.y_max.max(), self.df.z_max.max())
        return mins, maxs
    
    def normalize(self, tri, mins, maxs):
        """
        Normalizes the values in each row of tri according to mins, maxs

        Args:
            tri: np.array, MxN
            mins: np.array, 1xN
            maxs: np.array, 1xN
        
        Returns:
            Normalized MxN array
        """
        for i in len(tri):
            tri[i] = (tri[i] - mins[i]) / (maxs[i] - mins[i])
        return tri
    
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

    def batchmaker(self, batch_size, normalize=False, flat=False, pad_length=None, filenames=False):
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
        mins, maxs = self._prep_normalization()
        for i, stl_file in enumerate(self.df['stl_file']):
            stl_path = os.path.join(THINGI10K_STL_DIR, stl_file)
            vectors = read_mesh_vectors(stl_path)
            if normalize:
                for i, v in enumerate(vectors):
                    vectors[i] = self.normalize(v, mins, maxs)
            if pad_length:
                vectors = self._pad_vectors(vectors, pad_length)
            if flat:
                vectors = self._flatten_vectors(vectors)
            if filenames:
                batch.append((stl_path, vectors))
            else:
                batch.append(vectors)
            if (i+1) % batch_size == 0 or (i+1) == len(self.df):
                yield np.asarray(batch)
                batch = list()
        return
    
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
    
    def __getitem__(self, n):
        return self.df.loc[n]
    
    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return '<Thingi10k(index={}, n={}, pctile={})'.format(self.index, len(self), self.pctile)
    