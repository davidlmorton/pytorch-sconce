from sconce.data_feeds import ImageFeed

import pandas as pd


class MultiClassImageFeed(ImageFeed):
    """
    An ImageFeed class for use when each image may belong to more than one class.
    """
    def _get_class_df(self):
        dataset = self.dataset
        rows = []

        for target in self.get_targets():
            row = {_class: False for _class in dataset.classes}
            for idx in target:
                _class = dataset.classes[idx]
                row[_class] = True
            rows.append(row)

        return pd.DataFrame(rows)
