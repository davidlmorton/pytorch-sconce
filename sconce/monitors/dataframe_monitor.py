from matplotlib import pyplot as plt
from sconce.monitors.base import Monitor
from torch.autograd import Variable

import math
import matplotlib.patheffects as path_effects
import pandas as pd
import re
import stringcase


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
        self.previous_session_steps = 0
        self.last_step = 0

    def start_session(self, num_steps, **kwargs):
        self.previous_session_steps += self.last_step

    def is_blacklisted(self, key):
        for regex in self._blacklist_regexps:
            if regex.search(key):
                return True
        return False

    def write(self, data, step):
        data['step'] = step + self.previous_session_steps
        data['timestamp'] = pd.Timestamp.now()

        formatted_data = {}
        for k, v in data.items():
            if self.is_blacklisted(k):
                continue

            if isinstance(v, Variable):
                v = v.data[0]

            formatted_data[k] = v

        self._buffered_data.append(formatted_data)
        self.last_step = math.ceil(step)

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
                    metrics=['loss'],
                    test_color='tomato',
                    training_color='mediumseagreen',
                    learning_rate_color='dodgerblue',
                    fig=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)
            metrics_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, fig=fig)
            lr_ax = plt.subplot2grid((4, 1), (3, 0), fig=fig)
        else:
            metrics_ax = fig.axes[0]
            lr_ax = fig.axes[1]

        df = self.df.loc[skip_first:]

        for i, metric in enumerate(metrics):
            training_df = df['training_%s' % metric]
            training_df.interpolate().plot(ax=metrics_ax,
                    color=training_color, label='', alpha=0.35)

            test_key = 'test_%s' % metric
            if test_key in df:
                test_df = df[test_key]
                test_df.interpolate().plot(ax=metrics_ax,
                        color=test_color, label='', alpha=0.35)

            training_smooth_df = training_df.rolling(smooth_window,
                    min_periods=1).mean()
            my_path_effects = [
                    path_effects.SimpleLineShadow(offset=(0, 0),
                        linewidth=5, alpha=0.2),
                    path_effects.SimpleLineShadow(linewidth=5, alpha=0.2),
                    path_effects.Normal()]
            training_smooth_df.interpolate().plot(ax=metrics_ax, linewidth=4,
                    color=training_color, path_effects=my_path_effects,
                    label='Training')

            if test_key in df:
                test_smooth_df = test_df.rolling(smooth_window,
                        min_periods=1).mean()
                test_smooth_df.interpolate().plot(ax=metrics_ax, linewidth=3,
                        color=test_color, path_effects=my_path_effects,
                        label='Test')

            metrics_ax.set_ylabel(stringcase.titlecase(metric))
            if i == 0:
                metrics_ax.grid(axis='y')
                if len(metrics) == 2:
                    metrics_ax.legend(loc='center right')
                    metrics_ax = metrics_ax.twinx()
                else:
                    metrics_ax.legend(loc='best')

        metrics_ax.set_title(title)

        df['learning_rate'].fillna(method='backfill').plot(ax=lr_ax,
                color=learning_rate_color,
                linewidth=3)
        max_lr = df['learning_rate'].max()
        lr_ax.set_ylabel(f'Learning Rate\n[max={max_lr}]')

        plt.tight_layout()
        return fig

    def plot_learning_rate_survey(self, ax=None, figure_kwargs={},
            **plot_kwargs):
        if ax is None:
            fig = plt.figure(**figure_kwargs)
            ax = fig.add_subplot(1, 1, 1)

        df = self.df.groupby(
                self.df['learning_rate'].fillna(method='backfill')).mean()
        ax.loglog(df['learning_rate'], df['training_loss'], **plot_kwargs)
        ax.set_xlabel('Learning Rate (logscale)')
        ax.set_ylabel('Loss (logscale)')
        ax.set_title('Learning Rate Survey')

        return ax
