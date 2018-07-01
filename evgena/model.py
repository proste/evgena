import os
import json
import inspect
import datetime
from types import SimpleNamespace
from collections import namedtuple
from abc import ABC, abstractmethod
from typing import Callable, Tuple, NewType, Sequence

import numpy as np
import tensorflow as tf
import keras

from evgena.dataset import Dataset
from evgena.metrics import ConfusionMatrix
from evgena.data_transformations import images_to_BHWC, shape_to_BHWC, decode_labels


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


class TfModel(Model):
    @classmethod
    def from_checkpoint(
        cls, path: str, batch_size: int = 32,
        inputs_collection: str = 'end_points/inputs',
        scores_collection: str = 'end_points/scores',
        is_training_collection: str = 'end_points/is_training'
    ) -> 'TfModel':
        """Constructs tensorflow model from checkpoint

        path : str
            path to .meta model checkpoint file
        batch_size : int
            batch_size to be used during inference
        inputs_collection, scores_collection : str
            name of collection containing input/scores tensor;
            the first element of each collection will be taken
        is_training_collection : str, optional
            name of collection containing training phase tensor;
            None if no such tensor exists

        Returns
        -------
        TfModel
            loaded model

        """
        session = tf.Session(
            graph=tf.Graph(),
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True
                )
            )
        )

        with session.graph.as_default():
            saver = tf.train.import_meta_graph(path + '.meta')
            saver.restore(session, path)

        inputs, scores = [
            session.graph.get_collection(collection)[0]
            for collection in [
                inputs_collection, scores_collection
            ]
        ]

        if is_training_collection is None:
            is_training = None
        else:
            is_training = session.graph.get_collection(is_training_collection)[0]

        return cls(batch_size, session, inputs, scores, is_training)

    def __init__(
        self, batch_size: int, session: tf.Session, inputs: tf.Tensor,
        scores: tf.Tensor, is_training: tf.Tensor = None
    ):
        super(TfModel, self).__init__()

        self._session = session
        self.__batch_size = batch_size
        self.__inputs = inputs
        self.__scores = scores
        self.__is_training = is_training

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        inputs = images_to_BHWC(inputs)

        results = []

        feed_dict = {} if (self.__is_training is None) else {self.__is_training: False}
        for batch_begin in range(0, len(inputs), self.__batch_size):
            batch_end = batch_begin + self.__batch_size

            feed_dict[self.__inputs] = inputs[batch_begin:batch_end]

            results.append(self._session.run(
                self.__scores, feed_dict=feed_dict
            ))

        return np.concatenate(results)


Network = namedtuple('Network', [
    'learning_rate', 'is_training', 'global_step', 'images', 'labels', 'norm_labels',
    'label_weights', 'logits', 'predictions', 'scores', 'loss', 'training', 'summary_init',
    'curr_acc', 'update_acc', 'curr_loss', 'update_loss', 'reset_metrics',
    'flush_summaries', 'train_summaries', 'val_summaries', 'test_summaries'
])


ModelConfig = namedtuple('ModelConfig', [
    'inference_batch_size', 'batch_size', 'learning_rate', 'seed', 'constructor',
    'optimizer', 'dataset_path', 'do_shuffle', 'do_stratified', 'moment_axis'
])


ModelConstructor = NewType('ModelConstructor', Callable[
    [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
    Tuple[tf.Tensor, tf.Tensor]
])

OptimizerConstructor = NewType('OptimizerConstructor', Callable[
    [tf.Tensor], tf.train.Optimizer
])

# needs to be called - lazy loading issues
tf.contrib.summary


class TrainableTfModel(TfModel):
    @classmethod
    def construct(
        cls, constructor: ModelConstructor, dataset_path: str, batch_size: int,
        learning_rate: float, do_shuffle: bool = True, do_stratified: bool = True,
        optimizer: OptimizerConstructor = lambda lr: tf.train.AdamOptimizer(learning_rate=lr),
        moment_axis: Sequence[int] = None, weight_decay: float = None, seed: int = 42,
        tag: str = '', inference_batch_size: int = None
    ):
        # load dataset and get count of labels, optionally dataset moments
        dataset = Dataset.from_nprecord(dataset_path)
        labels_count = max(
            (split.y.max() + 1) if (split.y.ndim == 1) else split.y.shape[1]
            for split in (dataset.train, dataset.val, dataset.test)
        )
        labels, frequencies = np.unique(decode_labels(dataset.val.y), return_counts=True)
        label_frequencies = np.zeros(shape=labels_count, dtype=np.float32)
        label_frequencies[labels] = frequencies
        weights = len(dataset.val.y) / (label_frequencies * labels_count)

        if moment_axis is None:
            input_moments = None
        else:
            input_moments = np.mean(dataset.train.X, axis=moment_axis), np.std(dataset.train.X, axis=moment_axis)

        # prepare graph and session
        graph = tf.Graph()
        graph.seed = seed
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(allow_growth=True)
        )
        session = tf.Session(
            graph=graph,
            config=config
        )

        # deduce model name and paths
        name = '.'.join((
            datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'),
            'bs-{:04d}'.format(batch_size),
            'lr-{:.4f}'.format(learning_rate),
            'seed-{}'.format(seed)
        ))
        model_dir = os.path.join('models', tag, name)
        log_dir = os.path.join('logs', 'tf', tag, name)

        # construct network
        network = SimpleNamespace()
        with session.graph.as_default():
            network.global_step = tf.train.create_global_step()

            # inputs
            network.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
            network.is_training = tf.placeholder(tf.bool, [], name='is_training')
            network.images = tf.placeholder(
                tf.float32, shape=[None, *shape_to_BHWC(dataset.train.X.shape)[1:]], name='images'
            )
            network.labels = tf.placeholder(
                tf.float32, name='labels'
            )

            # label format normalization
            network.norm_labels = tf.cond(
                tf.equal(tf.rank(network.labels), 1),
                true_fn=(lambda: tf.one_hot(tf.to_int32(network.labels), labels_count)),
                false_fn=(lambda: network.labels),
                name='norm_labels'
            )
            network.label_weights = tf.squeeze(
                tf.matmul(network.norm_labels, tf.expand_dims(weights, axis=-1))
            )

            # input normalization
            x = network.images
            if input_moments is not None:
                mean, std = input_moments
                x = tf.divide(tf.subtract(x, mean), std)

            # network core definition
            with tf.variable_scope('core', regularizer=(None if (weight_decay is None) else tf.nn.l2_loss)):
                last_layer = constructor(
                    x, network.norm_labels, network.is_training, network.global_step
                )

                network.logits = tf.layers.dense(last_layer, labels_count)

            # outputs
            network.predictions = tf.argmax(network.logits, axis=-1, output_type=tf.int32, name='predictions')
            network.scores = tf.nn.softmax(network.logits, name='scores')

            # loss
            network.loss = tf.losses.softmax_cross_entropy(
                network.norm_labels, network.logits,
                weights=network.label_weights
            )
            if weight_decay is None:
                total_loss = network.loss
            else:
                total_loss = network.loss + weight_decay * tf.reduce_sum(tf.losses.get_regularization_losses())

            # training
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                network.training = optimizer(network.learning_rate).minimize(
                    total_loss, global_step=network.global_step, name='training'
                )

            # summaries
            network.curr_acc, network.update_acc = tf.metrics.accuracy(
                tf.argmax(network.norm_labels, axis=-1, output_type=tf.int32), network.predictions
            )
            network.curr_loss, network.update_loss = tf.metrics.mean(
                network.loss, weights=tf.shape(network.images)[0]
            )
            network.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            os.makedirs(log_dir, exist_ok=True)
            summary_writer = tf.contrib.summary.create_file_writer(
                log_dir, flush_millis=(10 * 1000)
            )

            with summary_writer.as_default():
                network.summary_init = tf.contrib.summary.summary_writer_initializer_op()
                network.flush_summaries = tf.contrib.summary.flush()

                with tf.contrib.summary.record_summaries_every_n_global_steps(100):
                    network.train_summaries = (
                        tf.contrib.summary.scalar('train/loss', network.update_loss),
                        tf.contrib.summary.scalar('train/accuracy', network.update_acc)
                    )

                with tf.contrib.summary.always_record_summaries():
                    network.val_summaries, network.test_summaries = [
                        (tf.contrib.summary.scalar(dataset + '/loss', network.curr_loss),
                         tf.contrib.summary.scalar(dataset + '/accuracy', network.curr_acc))
                        for dataset in ['val', 'test']
                    ]

            # Saver
            os.makedirs(model_dir, exist_ok=False)
            network = Network(**vars(network))
            for end_point_name, end_points in network._asdict().items():
                for end_point in (end_points if hasattr(end_points, '__len__') else [end_points]):
                    tf.add_to_collection(
                        'end_points/{}'.format(end_point_name),
                        end_point
                    )

            saver = tf.train.Saver(max_to_keep=None)

            # Variable initialization
            session.run((tf.global_variables_initializer(), network.summary_init))

            saver.save(session, os.path.join(model_dir, 'last'))

        if inference_batch_size is None:
            inference_batch_size = batch_size

        config = ModelConfig(
            inference_batch_size=inference_batch_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            constructor=inspect.getsource(constructor),
            optimizer=inspect.getsource(optimizer),
            dataset_path=dataset_path,
            do_shuffle=do_shuffle,
            do_stratified=do_stratified,
            moment_axis=moment_axis
        )

        with open(os.path.join(model_dir, 'config.json'), 'w') as config_f:
            json.dump(config._asdict(), config_f)

        with open(os.path.join(model_dir, 'training_log.json'), 'w') as log_f:
            json.dump({}, log_f)

        return cls(model_dir)

    @classmethod
    def load(
        cls, model_dir: str
    ):
        """
        - factories
            - new model from constructor, dataset, seed, tag
            - load model from path
        - init - just register variables

        - two batch_sizes
            - inference batch size - ie. [inputs] -> [scores, predictions]
            - back_prop batch size - ie. [inputs, labels] -> [loss, training]

        - factory
            - load existing model, with (un)available tensors
        - init for private use

        """
        pass

    def __init__(self, model_dir: str, inference_batch_size: int = None):
        self._model_dir = model_dir

        # load config and dataset
        with open(os.path.join(self._model_dir, 'config.json'), 'r') as config_f:
            self._config = ModelConfig(**json.load(config_f))

            if inference_batch_size is not None:
                self._config.inference_batch_size = inference_batch_size

            self._dataset = Dataset.from_nprecord(self._config.dataset_path)

        # load training log
        with open(os.path.join(self._model_dir, 'training_log.json'), 'r') as log_f:
            self._training_log = json.load(log_f)
            self._epochs = max((0, *map(int, self._training_log.keys())))

        # prepare graph and session
        session = tf.Session(
            graph=tf.Graph(),
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(allow_growth=True)
            )
        )

        # restore variables and network handles; create gradients
        with session.graph.as_default():
            self._saver = tf.train.import_meta_graph(os.path.join(self._model_dir, 'last.meta'))
            self._saver.restore(session, os.path.join(self._model_dir, 'last'))

            network_kwargs = {}
            for coll_name in Network._fields:
                coll = session.graph.get_collection('end_points/' + coll_name)
                network_kwargs[coll_name] = coll[0] if (len(coll) == 1) else coll
            self._network = Network(**network_kwargs)

            self._gradients = tf.gradients(self._network.loss, self._network.images)

            session.run(self._network.summary_init)

        # construct base class
        super().__init__(
            self._config.inference_batch_size, session, self._network.images,
            self._network.scores, self._network.is_training
        )

    def restore(self, ckpt_name: str = 'last'):
        self._saver.restore(self._session, os.path.join(self._model_dir, ckpt_name))

    def _save(self, ckpt_name: str = 'last'):
        self._saver.save(
            self._session, os.path.join(self._model_dir, ckpt_name),
            write_meta_graph=False
        )

    def gradients(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        inputs = images_to_BHWC(inputs)
        results = []

        for batch_begin in range(0, len(inputs), self._config.batch_size):
            batch_end = batch_begin + self._config.batch_size

            results.append(self._session.run(self._gradients, feed_dict={
                self._network.is_training: False,
                self._network.images: inputs[batch_begin:batch_end],
                self._network.labels: labels[batch_begin:batch_end]
            })[0])

        return np.concatenate(results)

    def _train_epoch(self, learning_rate: float):
        for batch in self._dataset.batch_over_train(
            self._config.batch_size, self._config.do_shuffle, self._config.do_stratified
        ):
            self._session.run(self._network.reset_metrics)
            self._session.run(
                [self._network.training, self._network.train_summaries],
                feed_dict={
                    self._network.images: images_to_BHWC(batch.X),
                    self._network.labels: batch.y,
                    self._network.is_training: True,
                    self._network.learning_rate: learning_rate
                }
            )

        self._epochs += 1

    def _evaluate(self, split_name, write_summaries=True):
        if split_name == 'val':
            batch_generator = self._dataset.batch_over_val
            split_summaries = self._network.val_summaries
            gold_labels = decode_labels(self._dataset.val.y)
        elif split_name == 'test':
            batch_generator = self._dataset.batch_over_test
            split_summaries = self._network.test_summaries
            gold_labels = decode_labels(self._dataset.test.y)
        else:
            raise ValueError('Invalid dataset split name')

        predictions = []
        self._session.run(self._network.reset_metrics)
        for batch in batch_generator(self._config.inference_batch_size):
            batch_predictions, *_ = self._session.run(
                [
                    self._network.predictions,
                    self._network.update_acc,
                    self._network.update_loss
                ], feed_dict={
                    self._network.images: images_to_BHWC(batch.X),
                    self._network.labels: batch.y,
                    self._network.is_training: False
                }
            )

            predictions.append(batch_predictions)

        acc, loss = self._session.run([self._network.curr_acc, self._network.curr_loss])

        if write_summaries:
            self._session.run(split_summaries)

        confusion_matrix = ConfusionMatrix(
            np.stack((gold_labels, np.concatenate(predictions)), axis=1),
            self._dataset.metadata.get('synset', None)
        )

        return acc, loss, confusion_matrix

    def train(self, epochs: int, from_checkpoint: str = 'last', learning_rate: float = None):
        if learning_rate is None:
            learning_rate = self._config.learning_rate

        # restore weights
        self.restore(from_checkpoint)

        # restore best metrics
        best_acc = max(0, 0, *(
            entry['best_acc'] for entry in self._training_log.values()
        ))
        best_loss = min(666, 666, *(
            entry['best_loss'] for entry in self._training_log.values()
        ))
        for e_i in range(epochs):
            self._train_epoch(learning_rate)

            val_acc, val_loss, _ = self._evaluate('val')
            print('Epoch: {e:02d}; val acc {acc:.4f}; val loss {loss:.4f}'.format(
                e=self._epochs, acc=val_acc, loss=val_loss
            ))

            if val_loss < best_loss:
                self._save('best_loss')
                best_loss = val_loss

            if val_acc > best_acc:
                self._save('best_acc')
                best_acc = val_acc

        # save last reached state
        self._save('last')

        # log training
        self._training_log[self._epochs] = {
            'best_loss': float(best_loss),
            'best_acc': float(best_acc),
            'learning_rate': learning_rate,
            'from_checkpoint': from_checkpoint
        }

        with open(os.path.join(self._model_dir, 'training_log.json'), 'w') as log_f:
            json.dump(self._training_log, log_f)

        print('Best loss: {l}; Best acc: {a}'.format(l=best_loss, a=best_acc))

        return best_loss

    def evaluate(self, checkpoint: str = 'last'):
        self.restore(checkpoint)
        acc, loss, confusion_matrix = self._evaluate('test', write_summaries=False)

        return acc, loss, confusion_matrix
