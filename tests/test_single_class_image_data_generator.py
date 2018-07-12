from sconce.data_generators import SingleClassImageDataGenerator

import os
import pandas as pd
import unittest

THIS_DIR = os.path.dirname(__file__)


class TestSingleClassDataGenerator(unittest.TestCase):
    def test_from_folder(self):
        dg = SingleClassImageDataGenerator.from_image_folder(
                root=os.path.join(THIS_DIR, 'data', 'image_folder'))

        df = dg.get_class_df()

        expected_df = pd.DataFrame([
            dict(cats=True, dogs=False, toys=False),
            dict(cats=False, dogs=True, toys=False),
            dict(cats=False, dogs=False, toys=True),
            dict(cats=False, dogs=False, toys=True),
        ])
        self.assertTrue(df.equals(expected_df))
