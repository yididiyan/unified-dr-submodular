from abc import abstractmethod, ABC
class Function(ABC):
    @abstractmethod
    def value(self, x):
        pass

    @abstractmethod
    def gradient_exact(self, x):
        pass

    @abstractmethod
    def gradient_noisy(self, x):
        pass



class CompositeFunction(ABC):

    @abstractmethod
    def value_individual(self, x, t):
        pass

    @abstractmethod
    def value_sum(self, x, t):
        pass
    @abstractmethod
    def gradient_exact_individual(self, x, t):
        pass
    @abstractmethod
    def gradient_exact_sum(self, x, t):
       pass
    @abstractmethod
    def gradient_noisy_individual(self, x, t):
        pass
    def gradient_noisy_sum(self, x, t):
        pass

