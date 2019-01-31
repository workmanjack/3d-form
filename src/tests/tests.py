# project imports
import env
from make_dataset import DATA_DIR, THINGI10K_INDEX


# python packages
import unittest
import os


class TestMakeDataset(unittest.TestCase):

    def test_DATA_DIR(self):
        os.path.exists(DATA_DIR)

    def test_THINGI10K_INDEX(self):
        os.path.exists(os.path.dirname(THINGI10K_INDEX))
