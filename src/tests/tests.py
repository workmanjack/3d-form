# project imports
import env
from data import DATA_DIR, THINGI10K_INDEX, PROCESSED_DIR
from data.stl import read_mesh_vectors
from data.thingi10k import thingi10k_batch_generator


# python packages
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

    def test_thingi10k_batch_generator_even_batches(self):
        batches = list()
        for batch in thingi10k_batch_generator(TEST_INDEX, 5):
            batches.append(batch)
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 5)
        self.assertEqual(len(batches[1]), 5)

    def test_thingi10k_batch_generator_uneven_batches(self):
        batches = list()
        for batch in thingi10k_batch_generator(TEST_INDEX, 3):
            batches.append(batch)
        self.assertEqual(len(batches), 4)
        self.assertEqual(len(batches[0]), 3)
        self.assertEqual(len(batches[1]), 3)
        self.assertEqual(len(batches[2]), 3)
        self.assertEqual(len(batches[3]), 1)


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
