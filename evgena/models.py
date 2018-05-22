from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import keras

from evgena.data_transformations import images_to_BHWC


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
    def __init__(
        self, path: str, batch_size: int = 32,
        inputs_collection: str = 'end_points/inputs',
        scores_collection: str = 'end_points/scores',
        predictions_collection: str = 'end_points/predictions',
        labels_collection: str = 'end_points/labels',
        loss_collection: str = 'end_points/loss',
        is_training_collection: str = 'end_points/is_training'
    ):
        super(TfModel, self).__init__()
        
        self._batch_size = batch_size
        self._graph = tf.Graph()
        self._session = tf.Session(
            graph=self._graph,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True
                )
            )
        )
        
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(self._session, path)
        
        self._inputs, self._scores, self._predictions, self._labels, self._loss, self._is_training = [
            (self._graph.get_collection(collection) + [None])[0]
            for collection in [
                inputs_collection, scores_collection, predictions_collection,
                labels_collection, loss_collection, is_training_collection
            ]
        ]
        
        if self._loss is None:
            self._gradients = None
        else:
            self._gradients = tf.gradients(self._loss, self._inputs)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs = images_to_BHWC(inputs)
        
        results = []
        
        for batch_begin in range(0, len(inputs), self._batch_size):
            batch_end = batch_begin + self._batch_size
            
            results.append(self._session.run(
                self._scores, feed_dict={
                    self._is_training: False,
                    self._inputs: inputs[batch_begin:batch_end]
                }
            ))
        
        return np.concatenate(results)
    
    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        inputs = images_to_BHWC(inputs)
        
        if self._gradients is None:
            return None
        else:
            return self._session.run(self._gradients, feed_dict={
                self._is_training: False,
                self._inputs: inputs,
                self._labels: labels
            })[0]
