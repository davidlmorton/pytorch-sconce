from abc import ABC, abstractmethod


class RateController(ABC):
    @abstractmethod
    def start_session(self, num_steps):
        pass

    @abstractmethod
    def new_learning_rate(self, step, data):
        pass
