from sconce.data_feeds import SingleClassImageFeed

import os
import pandas as pd
import unittest

THIS_DIR = os.path.dirname(__file__)


class TestSingleClassImageFeed(unittest.TestCase):
    def test_from_folder(self):
        feed = SingleClassImageFeed.from_image_folder(
                root=os.path.join(THIS_DIR, 'data', 'image_folder'))

        df = feed.get_class_df()

        expected_df = pd.DataFrame([
            dict(cats=True, dogs=False, toys=False),
            dict(cats=False, dogs=True, toys=False),
            dict(cats=False, dogs=False, toys=True),
            dict(cats=False, dogs=False, toys=True),
        ])
        self.assertTrue(df.equals(expected_df))
