from sconce.monitors.base import Monitor
from torch.autograd import Variable
import losswise


class LosswiseMonitor(Monitor):
    def __init__(self, api_key, tag,
            params={},
            min_graphs={
                'loss': {
                    'training_loss': 'Training Loss',
                    'test_loss': 'Test Loss',
                },
                'lr': {
                    'learning_rate': 'Learning Rate',
                }},
            max_graphs={},
            name='losswise_monitor'):
        super().__init__(name=name)

        losswise.set_api_key(api_key)
        self._session = losswise.Session(tag=tag, params=params,
                track_git=False)

        self._min_graphs = min_graphs
        self._max_graphs = max_graphs

        self._graphs = {}
        for tracker in min_graphs:
            self._graphs[tracker] = self._session.graph(tracker, kind='min')
        for tracker in max_graphs:
            self._graphs[tracker] = self._session.graph(tracker, kind='max')

        self.step_num = 0

    def start_session(self, num_steps):
        pass

    @property
    def _graph_descriptions(self):
        return {**self._min_graphs, **self._max_graphs}

    def step(self, data):
        self.step_num += 1

        for tracker, metrics in self._graph_descriptions.items():
            graph = self._graphs[tracker]

            values = {}
            for key, name in metrics.items():
                if key in data:
                    value = data[key]
                    if isinstance(value, Variable):
                        value = value.data[0]
                    values[key] = value
            graph.append(self.step_num, values)
