from sconce.data_feeds import ImageFeed

import pandas as pd


class MultiClassImageFeed(ImageFeed):
    """
    An ImageFeed class for use when each image may belong to more than one class.
    """
    def _get_class_df(self, targets=None):
        dataset = self.dataset
        rows = []

        if targets is None:
            if hasattr(dataset, 'targets'):
                targets = dataset.targets
            else:
                raise RuntimeError("No targets were supplied, and the dataset doesn't "
                                   "have a 'targets' attribute")

        for target in targets:
            row = {_class: False for _class in dataset.classes}
            for idx in target:
                _class = dataset.classes[idx]
                row[_class] = True
            rows.append(row)

        return pd.DataFrame(rows)
