# project imports
import env
from data import DATA_DIR, THINGI10K_INDEX, PROCESSED_DIR
from data.stl import read_mesh_vectors
from data.thingi10k import Thingi10k
from utils import PROJECT_ROOT
from models import MODEL_DIR


# python packages
import pandas as pd
import numpy as np
import unittest
import os


YODA_STL = os.path.join(PROCESSED_DIR, '37861_yoda.stl')
OCTOCAT_STL = os.path.join(PROCESSED_DIR, '32770_octocat.stl')
CONTENT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), 'content'))
TEST_STL = os.path.join(CONTENT_DIR, '{}_test.stl'.format(100027))
TEST_INDEX = os.path.join(CONTENT_DIR, 'thingi10k_index_10.csv')


class TestPaths(unittest.TestCase):

    def test_DATA_DIR(self):
        self.assertTrue(os.path.exists(DATA_DIR))

    def test_THINGI10K_INDEX(self):
        self.assertTrue(os.path.exists(os.path.dirname(THINGI10K_INDEX)))

    def test_PROJECT_ROOT(self):
        self.assertTrue(os.path.exists(PROJECT_ROOT))

    def test_MODEL_DIR(self):
        self.assertTrue(os.path.exists(MODEL_DIR))

        
class TestThingi10k(unittest.TestCase):
    
    def setUp(self):
        self.Thingi = Thingi10k.initFromIndex(TEST_INDEX)    

    def test_batchmaker_even_batches(self):
        batches = list()
        for batch in self.Thingi.batchmaker(5):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 5)
        self.assertEqual(len(batches[1]), 5)

    def test_batchmaker_uneven_batches(self):
        batches = list()
        for batch in self.Thingi.batchmaker(3):
            batches.append(batch)
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 3)
        self.assertEqual(len(batches[2]), 3)
        self.assertEqual(len(batches[3]), 1)

    def test_batchmaker_flat_true_pad_length_none(self):
        batches = list()
        for batch in self.Thingi.batchmaker(5, flat=True):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        # len value taken from manual inspection of csv num_faces
        self.assertEqual(len(batches[0][0]), 10746*3*3)
        # spot check that first and last vertices are not zero
        # to ensure padding has not been applied
        self.assertTrue(batches[0][0][0] != 0)
        self.assertTrue(batches[0][0][1] != 0)
        self.assertTrue(batches[0][0][2] != 0)
        self.assertTrue(batches[0][0][-1] != 0)
        self.assertTrue(batches[0][0][-2] != 0)
        self.assertTrue(batches[0][0][-3] != 0)

    def test_batchmaker_flat_true_pad_length_given(self):
        batches = list()
        pad_length = 90000
        for batch in self.Thingi.batchmaker(5, flat=True, pad_length=pad_length):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0][0]), pad_length)
        self.assertEqual(len(batches[1][2]), pad_length)

    def test_batchmaker_flat_true_pad_length_truncate(self):
        batches = list()
        pad_length = 9
        for batch in self.Thingi.batchmaker(5, flat=True, pad_length=pad_length):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0][0]), pad_length)
        self.assertEqual(len(batches[1][2]), pad_length)

    def test_batchmaker_normalize_true(self):
        batches = list()
        pad_length = 9
        for batch in self.Thingi.batchmaker(5, normalize=True, filenames=True):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            for stl_file, vectors in batch:
                self.assertTrue(vectors.max() <= 1)
                self.assertTrue(vectors.min() >= 0)

    def test_batchmaker_normalize_true_triangles_true(self):
        batches = list()
        pad_length = 9
        for batch in self.Thingi.batchmaker(5, normalize=True, triangles=True):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            for vectors in batch:
                self.assertTrue(vectors.max() <= 1)
                self.assertTrue(vectors.min() >= 0)
                self.assertTrue(vectors.shape == (len(vectors), 9))
                
    def test__prep_normalization(self):
        # values taken from manual inspection of csv num_faces
        actual_mins, actual_maxs = self.Thingi._prep_normalization()
        expected_mins = [-83.2, -63.5, -213.192]
        expected_maxs = [51.16601, 63.5, 121]
        self.assertTrue(expected_mins, actual_mins)
        self.assertTrue(expected_maxs, actual_maxs)
        
    def test__pad_vectors_append(self):
        vectors = np.ones([99, 3, 3])
        pad_length = 100*3*3
        flat = False
        expected = np.concatenate((vectors, np.zeros([1, 3, 3])), axis=0)
        actual = self.Thingi._pad_vectors(vectors=vectors, pad_length=pad_length)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(actual[-1].sum(), 0)
        self.assertEqual(actual[0].sum(), 9.0)
        
    def test__pad_vectors_truncate(self):
        vectors = np.ones([101, 3, 3])
        pad_length = 100*3*3
        flat = False
        expected = np.ones([100, 3, 3])
        actual = self.Thingi._pad_vectors(vectors=vectors, pad_length=pad_length)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(actual[-1].sum(), 9.0)
        self.assertEqual(actual[0].sum(), 9.0)

    def test__pad_vectors_not_mult_9(self):
        vectors = np.ones([1, 3, 3])
        pad_length = 10
        expected = vectors
        actual = self.Thingi._pad_vectors(vectors=vectors, pad_length=pad_length)
        self.assertEqual(expected.shape, actual.shape)

    def test__flatten_vectors_n_dim(self):
        # use specific numbers to validate that numbers are reshaped into expected spots
        vectors = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])
        expected = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9])
        actual = self.Thingi._flatten_vectors(vectors)
        self.assertEqual(expected.shape, actual.shape)

    def test__flatten_vectors_one_dim(self):
        vectors = np.ones([5])
        expected = np.ones([5])
        actual = self.Thingi._flatten_vectors(vectors)
        self.assertEqual(expected.shape, actual.shape)

    def test__triangulize_vectors_n_dim(self):
        vectors = np.asarray([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ], [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]
            ]
        ])
        expected = np.asarray([
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18]
        ])
        actual = self.Thingi._triangulize_vectors(vectors)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(np.array_equal(expected, actual))
        
    def test__reform_vectors_n_dim(self):
        vectors = np.random.rand(5, 3, 3)
        expected = vectors
        actual = self.Thingi._reform_vectors(vectors)
        self.assertEqual(expected.shape, actual.shape)

    def test__reform_vectors_one_dim(self):
        vectors = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                            10, 11, 12, 13, 14, 15, 16, 17, 18])
        expected = np.array([[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]],
            [[10, 11, 12],
             [13, 14, 15],
             [16, 17, 18]]])
        actual = self.Thingi._reform_vectors(vectors)
        self.assertEqual(expected.shape, actual.shape)

    def test__normalize_vectors(self):
        vectors = np.asarray([[[0, 1, 2], [0, 5, 10], [0, 10, 20]],
                               [[1, 0, 2], [1, 4, 6], [4, 8, 16]]])
        xyz_min = 0
        xyz_max = 20
        expected = np.asarray([[[0, .05, .1], [0, .25, .5], [0, .5, 1.0]],
                               [[.05, 0, .1], [.05, .2, .3], [.2, .4, .8]]])
        actual = self.Thingi._normalize_vectors(vectors, xyz_min, xyz_max)
        self.assertTrue(np.array_equal(expected, actual))
        
        
class TestStl(unittest.TestCase):

    def test_read_mesh_vectors_file_dne(self):
        expected = None
        actual = read_mesh_vectors('banana')
        self.assertEqual(expected, actual)

    def test_read_mesh_vectors_file_not_stl(self):
        expected = None
        actual = read_mesh_vectors('banana.txt')
        self.assertEqual(expected, actual)

    def test_read_mesh_vectors_file_exists(self):
        expected = np.asarray([
            [-775.281421038229, -68.2960126417953, -444.709630352384],
            [-772.985586951259, -58.981344042603, -444.709630303666],
            [-775.281421038229, -58.9252444105971, -444.709630352384]
        ])
        actual = read_mesh_vectors(TEST_STL)
        self.assertEqual(expected.all(), actual.all())
        