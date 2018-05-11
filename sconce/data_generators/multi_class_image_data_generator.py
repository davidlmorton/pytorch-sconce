from sconce.data_generators import DataGenerator, ImageMixin

import pandas as pd


class MultiClassImageDataGenerator(DataGenerator, ImageMixin):
    """
    An ImageDataGenerator class for use when each image may belong to more than one class.

    New in 0.10.0
    """
    def _get_class_df(self):
        dataset = self.real_dataset
        rows = []

        for target in dataset.targets:
            row = {_class: False for _class in dataset.classes}
            for idx in target:
                _class = dataset.classes[idx]
                row[_class] = True
            rows.append(row)

        return pd.DataFrame(rows)
