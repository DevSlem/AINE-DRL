from abc import abstractmethod

from torch import nn

class Algorithm() :

    @abstractmethod
    def train(self) :
        pass
    
    @abstractmethod
    def get_pdparam(self) :
        pass

    @abstractmethod
    def update_hyperparameters(self) :
        pass






