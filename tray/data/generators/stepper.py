from abc import ABC, abstractmethod


class BaseStepper(ABC):
    """A framework to evolve a DGP to the next step"""

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def get_iterator(self):
        return self.__iter__()
