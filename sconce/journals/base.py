from abc import ABC, abstractmethod


class Journal(ABC):
    @abstractmethod
    def record_step(self, data):
        pass
