from abc import ABC, abstractmethod


class Classifier(ABC):
    def __init__(self, data):
        self._set_attributes(data)

    @abstractmethod
    def _set_attributes(self, data):
        pass

    @abstractmethod
    def predict_all(self, image, ystart, ystop, scale, pix_per_step):
        pass



