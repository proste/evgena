import numpy as np
import tensorflow as tf
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


# needs to be called - lazy loading issues
tf.contrib.summary

class TfModel(Model):
    def __init__(self, path, inputs_collection, outputs_collection, batch_size: int = 32):
        super(TfModel, self).__init__()

        self._session = tf.Session(graph=tf.Graph())
        
        with self._session.graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(self._session, path)

        self._training_phase = self._session.graph.get_collection('end_points/training_phase')[0]
        self._input = self._session.graph.get_collection(inputs_collection)[0]
        self._output = self._session.graph.get_collection(outputs_collection)[0]
        self._batch_size = batch_size

    def __call__(self, examples: np.ndarray) -> np.ndarray:
        if len(examples.shape[1:]) == 2:
            examples = np.expand_dims(examples, -1)
            
        result = np.empty([len(examples)] + self._output.shape.as_list()[1:], dtype=np.float32)
        
        for batch_begin in range(0, len(examples), self._batch_size):
            batch_end = batch_begin + self._batch_size
            result[batch_begin:batch_end] = self._session.run(self._output, feed_dict={self._training_phase: False, self._input: examples[batch_begin:batch_end]})

        return result