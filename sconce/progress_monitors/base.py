from abc import ABC, abstractmethod


class ProgressMonitor(ABC):
    @abstractmethod
    def step(self, data):
        pass

    @abstractmethod
    def start_session(self, num_steps, **kwargs):
        pass
