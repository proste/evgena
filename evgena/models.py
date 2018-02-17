import numpy as np
import keras

from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __call__(self, examples: np.ndarray) -> np.ndarray:
        NotImplemented


class KerasModel(Model):
    def __init__(self, path, batch_size: int = 32):
        super(KerasModel, self).__init__()

        self._model = keras.models.load_model(path)  # TODO compile??
        self._batch_size = batch_size

    def __call__(self, examples: np.ndarray) -> np.ndarray:
        return self._model.predict(examples, self._batch_size)
