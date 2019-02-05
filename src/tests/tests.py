# project imports
import env
from data import DATA_DIR, THINGI10K_INDEX, PROCESSED_DIR
from data.stl import read_mesh_vectors
from data.thingi10k import Thingi10k


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


class TestMakeDataset(unittest.TestCase):

    def test_DATA_DIR(self):
        os.path.exists(DATA_DIR)

    def test_THINGI10K_INDEX(self):
        os.path.exists(os.path.dirname(THINGI10K_INDEX))


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
        pad_length = 50000
        for batch in self.Thingi.batchmaker(5, flat=True, pad_length=pad_length):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0][0]), pad_length)
        self.assertEqual(len(batches[1][2]), pad_length)

    def test_batchmaker_flat_true_pad_length_truncate(self):
        batches = list()
        pad_length = 1
        for batch in self.Thingi.batchmaker(5, flat=True, pad_length=pad_length):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0][0]), pad_length)
        self.assertEqual(len(batches[1][2]), pad_length)


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
        