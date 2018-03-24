from matplotlib import pyplot as plt
from sconce.monitors.base import Monitor
from torch.autograd import Variable

import pandas as pd
import re


class DataframeMonitor(Monitor):
    def __init__(self, df=None, metadata=None,
            blacklist=['._inputs', '._outputs'],
            name='dataframe_monitor'):
        super().__init__(name=name)
        if metadata is None:
            metadata = {}
            metadata['created_at'] = pd.Timestamp.now()
        self.metadata = metadata

        self._buffered_data = []
        self._df = df
        self._blacklist = blacklist
        self._blacklist_regexps = [re.compile(x) for x in blacklist]
        self.step_num = 0

    def is_blacklisted(self, key):
        for regex in self._blacklist_regexps:
            if regex.search(key):
                return True
        return False

    def step(self, data):
        data['step'] = self.step_num
        data['timestamp'] = pd.Timestamp.now()

        formatted_data = {}
        for k, v in data.items():
            if self.is_blacklisted(k):
                continue

            if isinstance(v, Variable):
                v = v.data[0]

            formatted_data[k] = v

        self._buffered_data.append(formatted_data)
        self.step_num += 1

    @property
    def df(self):
        if self._buffered_data:
            buffered_df = pd.DataFrame(self._buffered_data).set_index('step')
            self._buffered_data = []

            if self._df is None:
                self._df = buffered_df
            else:
                self._df = pd.concat([self._df, buffered_df])

        return self._df

    @classmethod
    def from_file(cls, filename, key):
        with pd.HDFStore(filename) as store:
            df = store[key]
            metadata = store.get_storer(key).attrs.metadata
        obj = cls(df=df, metadata=metadata)
        return obj

    def save(self, filename, key):
        with pd.HDFStore(filename) as store:
            store.put(key, self.df)
            store.get_storer(key).attrs.metadata = self.metadata

    def plot(self, title="Training History", figsize=(15, 5),
                    skip_first=100, smooth_window=50,
                    logscale_learning_rate=False,
                    logscale_loss=False,
                    fig=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)

        df = self.df.loc[skip_first:]

        ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, fig=fig)
        training_loss = df['training_loss'].rolling(smooth_window,
                min_periods=1, center=True).mean()
        training_loss.plot(ax=ax, color='mediumseagreen',
                           logy=logscale_loss,
                           label='Training Loss')

        if 'test_loss' in df:
            test_loss = df['test_loss'].rolling(smooth_window,
                    min_periods=1, center=True).mean()
            test_loss.interpolate().plot(ax=ax, color='tomato',
                    logy=logscale_loss,
                    label='Test Loss')
        ax.set_title(title)
        ax.grid(axis='y')
        ax.legend()

        ax = plt.subplot2grid((4, 1), (3, 0), fig=fig)
        df['learning_rate'].plot(ax=ax, color='dodgerblue',
                              logy=logscale_learning_rate,
                              label='Learning Rate')
        ax.legend()

        plt.tight_layout()
        return fig

    def plot_learning_rate_survey(self, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(1, 1, 1)

        ax.loglog(self.df['learning_rate'], self.df['training_loss'])
        ax.set_xlabel('Learning Rate (logscale)')
        ax.set_ylabel('Loss (logscale)')
        ax.set_title('Learning Rate Survey')

        return ax
