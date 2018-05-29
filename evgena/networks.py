import os
import json
import inspect
import datetime
from typing import Callable, Tuple

import numpy as np
import tensorflow as tf

from .dataset import Dataset
from .data_transformations import images_to_BHWC, shape_to_BHWC
from .metrics import ConfusionMatrix


class Network:
    def __init__(
        self, constructor: Callable[['Network'], str], dataset_path: str,
        batch_size: int, learning_rate: float,
        seed: int = 42, tag: str = ''
    ):
        self.dataset_path = dataset_path
        self.dataset = Dataset.from_nprecord(dataset_path)
        self.labels_count = max(
            (split.y.max() + 1) if (split.y.ndim == 1) else split.y.shape[1]
            for split in (self.dataset.train, self.dataset.val, self.dataset.test)
        )
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.constructor_code = inspect.getsource(constructor)
        self.name = '.'.join((
            datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'), tag
        ))
        self.training_log = {}
        self.epochs = 0
        
        logdir = 'logs/tf/' + self.name + '/'
        os.makedirs(logdir, exist_ok=True)
        
        graph = tf.Graph()
        graph.seed = self.seed
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        self.session = tf.Session(
            graph=graph,
            config=config
        )
        
        with self.session.graph.as_default():
            self.is_training = tf.placeholder(tf.bool, [], name='is_training')
            self.global_step = tf.train.create_global_step()
            
            batch_size = tf.Dimension(None)
            self.images = tf.placeholder(
                tf.float32, [batch_size, *shape_to_BHWC(self.dataset.train.X.shape)[1:]], name='images'
            )
            self.labels = tf.placeholder(
                tf.float32, [batch_size, self.labels_count], name='labels'
            )

            constructor(self)  # define self.logits
            
            self.predictions = tf.argmax(self.logits, axis=-1, output_type=tf.int32)
            self.scores = tf.nn.softmax(self.logits)
            
            self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits, scope='loss')
            
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                    self.loss, global_step=self.global_step, name='training'
                )
            
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(
                tf.argmax(self.labels, axis=-1, output_type=tf.int32), self.predictions
            )
            self.current_loss, self.update_loss = tf.metrics.mean(
                self.loss, weights=tf.shape(self.images)[0]
            )
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))
            
            summary_writer = tf.contrib.summary.create_file_writer(
                logdir, flush_millis=10 * 1000
            )
            self.summaries = {}
            with summary_writer.as_default():
                self.flush_summaries = tf.contrib.summary.flush()
                
                with tf.contrib.summary.record_summaries_every_n_global_steps(100):
                    self.summaries['train'] = [
                        tf.contrib.summary.scalar('train/loss', self.update_loss),
                        tf.contrib.summary.scalar('train/accuracy', self.update_accuracy)
                    ]
                    
                with tf.contrib.summary.always_record_summaries():
                    for dataset in ['val', 'test']:
                        self.summaries[dataset] = [
                            tf.contrib.summary.scalar(dataset + '/loss', self.current_loss),
                            tf.contrib.summary.scalar(dataset + '/accuracy', self.current_accuracy)
                        ]
                    
            # Saver
            tf.add_to_collection('end_points/is_training', self.is_training)
            tf.add_to_collection('end_points/inputs', self.images)
            tf.add_to_collection('end_points/labels', self.labels)
            tf.add_to_collection('end_points/scores', self.scores)
            tf.add_to_collection('end_points/predictions', self.predictions)
            tf.add_to_collection('end_points/loss', self.loss)
            self.saver = tf.train.Saver(max_to_keep=None)
            
            # Variable initialization
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)
    
    def _train_epoch(self, do_shuffle, do_stratified):
        for batch in self.dataset.batch_over_train(self.batch_size, do_shuffle, do_stratified):
            self.session.run(self.reset_metrics)
            self.session.run(
                [self.training, self.summaries['train']],
                {
                    self.images: images_to_BHWC(batch.X),
                    self.labels: self._decode_labels(batch.y),
                    self.is_training: True
                }
            )
        
        self.epochs += 1
            
    def _evaluate(self, split_name):
        if split_name == 'val':
            batch_generator = self.dataset.batch_over_val
            gold_labels = self._encode_labels(self.dataset.val.y)
        elif split_name == 'test':
            batch_generator = self.dataset.batch_over_test
            gold_labels = self._encode_labels(self.dataset.test.y)
        else:
            raise ValueError('Invalid dataset split name')
        
        predictions = []
        self.session.run(self.reset_metrics)            
        for batch in batch_generator(self.batch_size):
            batch_predictions, *_ = self.session.run(
                [self.predictions, self.update_accuracy, self.update_loss],
                {
                    self.images: images_to_BHWC(batch.X),
                    self.labels: self._decode_labels(batch.y),
                    self.is_training: False
                }
            )
            
            predictions.append(batch_predictions)

        acc, loss, *_ = self.session.run([
            self.current_accuracy, self.current_loss, self.summaries[split_name]
        ])
        
        confusion_matrix = ConfusionMatrix(
            np.stack((gold_labels, np.concatenate(predictions)), axis=1),
            self.dataset.metadata.get('synset', None)
        )

        return acc, loss, confusion_matrix
    
    def _decode_labels(self, labels: np.ndarray) -> np.ndarray:
        if labels.ndim == 1:
            decoded = np.zeros(shape=(len(labels), self.labels_count), dtype=np.float32)
            decoded[np.arange(len(labels)), labels] = 1
            return decoded
        elif labels.ndim == 2:
            return labels
        else:
            raise ValueError('Invalid labels shape: {}'.format(labels.shape))

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        if labels.ndim == 1:
            return labels
        elif (labels.ndim == 2) and (labels.shape[1] == self.labels_count):
            return np.argmax(labels, axis=-1)
        else:
            raise ValueError('Invalid labels shape: {}'.format(labels.shape))

    def train(
        self, epochs: int,
        do_shuffle: bool = True, do_stratified: bool = True
    ):
        model_prefix = 'models/' + self.name + '/'
        os.makedirs(model_prefix, exist_ok=True)
        
        if self.epochs == 0:
            np.random.seed = self.seed

            with open(model_prefix + 'config.json', 'w') as config_f:
                json.dump({
                    'batch_size': self.batch_size,
                    'learning_rate': self.learning_rate,
                    'seed': self.seed,
                    'constructor': self.constructor_code,
                    'dataset_path': self.dataset_path
                }, config_f)
        else:
            self.saver.restore(self.session, model_prefix + 'last')
            
        checkpoint_prefix = model_prefix + str(self.epochs + epochs) + '-'
        
        # Train
        best_acc = 0
        best_loss = 666
        for e_i in range(epochs):
            self._train_epoch(do_shuffle, do_stratified)
            
            dev_acc, dev_loss, _ = self._evaluate('val')
            print('Epoch: {e:02d}: val acc {a:.4f}'.format(e=self.epochs, a=dev_acc))
    
            if best_loss > dev_loss:
                self.saver.save(self.session, checkpoint_prefix + 'best_loss')
                best_loss = dev_loss

            if best_acc < dev_acc:
                self.saver.save(self.session, checkpoint_prefix + 'best_acc')
                best_acc = dev_acc
        
        self.saver.save(self.session, model_prefix + 'last')
        
        # Test
        self.saver.restore(self.session, checkpoint_prefix + 'best_acc')
        acc_test_acc, acc_test_loss, acc_confusion_matrix = self._evaluate('test')

        self.saver.restore(self.session, checkpoint_prefix + 'best_loss')
        loss_test_acc, loss_test_loss, loss_confusion_matrix = self._evaluate('test')
        self.session.run(self.flush_summaries)
        
        self.training_log[self.epochs] = {
            'do_shuffle': do_shuffle,
            'do_stratified': do_stratified,
            'acc_test_acc': float(acc_test_acc),
            'acc_test_loss': float(acc_test_loss),
            'loss_test_acc': float(loss_test_acc),
            'loss_test_loss': float(loss_test_loss),
        }

        with open(model_prefix + 'training_log.json', 'w') as log_file:
            json.dump(self.training_log, log_file)
        
        print('Test acc: {a:.4f}'.format(a=acc_test_acc))

        return acc_confusion_matrix, loss_confusion_matrix
