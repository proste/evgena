from typing import List, Tuple, Any

import numpy as np
import tensorflow as tf


class SSIM:
    def __init__(self, size=11, sigma=1.5, K1=0.01, K2=0.03, seed=42):
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph)
        
        with graph.as_default():
            # input placeholders
            self.images_x = tf.placeholder(tf.float32, [None, None, None, 1], name='images_x')  # TODO shapes??
            self.images_y = tf.placeholder(tf.float32, [None, None, None, 1], name='images_y')
            
            # weighting gaussian window
            window_half = size // 2
            x_data, y_data = np.mgrid[(- window_half):(window_half + 1),(- window_half):(window_half + 1)]

            g = np.exp(-((x_data**2 + y_data**2)/(2.0*sigma**2)))
            normed_g = g / np.sum(g)
            normed_g = normed_g.reshape(normed_g.shape + (1, 1))
            
            window = tf.constant(normed_g, dtype=tf.float32)
            
            C1 = tf.constant(K1**2, dtype=tf.float32)
            C2 = tf.constant(K2**2, dtype=tf.float32)
            C3 = tf.constant((K2**2) / 2, dtype=tf.float32)
            
            # ssim computation
            mu_x = tf.nn.conv2d(self.images_x, window, strides=[1, 1, 1, 1], padding='VALID')
            mu_x_sq = mu_x * mu_x
            sigma_x_sq = tf.abs(tf.nn.conv2d(self.images_x*self.images_x, window, strides=[1, 1, 1, 1], padding='VALID') - mu_x_sq)
            sigma_x = tf.sqrt(sigma_x_sq)
            
            mu_y = tf.nn.conv2d(self.images_y, window, strides=[1, 1, 1, 1], padding='VALID')
            mu_y_sq = mu_y * mu_y
            sigma_y_sq = tf.abs(tf.nn.conv2d(self.images_y*self.images_y, window, strides=[1, 1, 1, 1], padding='VALID') - mu_y_sq)
            sigma_y = tf.sqrt(sigma_y_sq)
            
            mu_xy = mu_x * mu_y
            sigma_xy = tf.abs(tf.nn.conv2d(self.images_x*self.images_y, window, strides=[1, 1, 1, 1], padding='VALID') - mu_xy)
            
            self.ssim_luminance = (2 * mu_xy + C1) / (mu_x_sq + mu_y_sq + C1)
            self.ssim_contrast = (2 * sigma_x * sigma_y + C2) / (sigma_x_sq + sigma_y_sq + C2)
            self.ssim_structure = (sigma_xy + C3) / (sigma_x * sigma_y + C3)
            self.ssim_map = self.ssim_luminance * self.ssim_contrast * self.ssim_structure
            self.ssim_metrics = tf.reduce_mean(self.ssim_map, axis=[1, 2, 3])

    def __call__(self, images_x, images_y):
        if images_y.shape != images_x.shape:
            raise ValueError('images_x and images_y shapes mismatch - shapes must be equal')
            
        if len(images_x.shape[1:]) == 2:
            images_x = np.expand_dims(images_x, -1)
            images_y = np.expand_dims(images_y, -1)
        
        return self.session.run(
            self.ssim_metrics,
            feed_dict={self.images_x: images_x, self.images_y: images_y}
        )
    

def mse(images_x, images_y):
    if images_y.shape != images_x.shape:
            raise ValueError('images_x and images_y shapes mismatch - shapes must be equal')
    
    return np.mean(np.square(images_x - images_y), axis=tuple(range(1, images_x.ndim)))


class ConfusionMatrix:
    @property
    def accuracy(self):
        """Accuracy on data

        Returns
        -------
        float
            accuracy on data

        """
        return self._absolute.trace() / self._absolute.sum()

    @property
    def absolute(self):
        """Confusion matrix

        Returns
        -------
        np.ndarray
            2D confusion matrix, gold x predictions (rows x columns)

        """
        return self._absolute

    @property
    def gold_wrt_predicted(self):
        """Matrix of probabilities P(gold|prediction)

        i.e., probability of given prediction being gold label 

        Returns
        -------
        np.ndarray
            2D array of P(gold|prediction), predictions x gold (rows x columns)

        """
        return self._gold_wrt_predicted

    @property
    def predicted_wrt_gold(self):
        """Matrix of probabilities P(prediction|gold)

        i.e., probability of given gold label being predicted

        Returns
        -------
        np.ndarray
            2D array of P(prediction|gold), gold x prediction (rows x columns)

        """
        return self._predicted_wrt_gold

    def __init__(self, evaluation: List[Tuple[Any, Any]], label_names: List[str] = None):
        """Constructs confussion matrix from list of (gold, predicted) pairs

        Parameters
        ----------
        evaluation : List[Tuple[Any, Any]]
            list of pairs (gold, predicted) of labels or IDs of labels
        label_names : List[str], optional
            optional list of names of labels aligned with label IDs in evaluation

        """
        self._evaluation = np.asarray(evaluation)

        if label_names is None:
            self._label_ids = np.unique(self._evaluation)
            self._label_names = list(self._label_ids)
        else:
            self._label_ids = np.arange(len(label_names))
            self._label_names = label_names

        self._absolute = np.array([
            [
                (self._evaluation[self._evaluation[:, 0] == gold_id][:, 1] == predicted_id).sum()
                for predicted_id in self._label_ids
            ]
            for gold_id in self._label_ids
        ])

        self._gold_wrt_predicted = (self._absolute / self._absolute.sum(axis=0))
        self._predicted_wrt_gold = self._absolute / np.expand_dims(self._absolute.sum(axis=1), axis=1)

    def _matrix_to_md(self, matrix, name, formatting):
        str_matrix = []
        str_matrix.append([name] + list(map(str, self._label_names)))
        for label_name, row in zip(self._label_names, matrix):
            str_matrix.append([str(label_name)] + [('{:' + formatting + '}').format(item) for item in row])

        column_widths = list(map(max, zip(*[map(len, row) for row in str_matrix])))

        formatted_matrix = [
            [
                ('{: >' + str(column_width) + '}').format(item)
                for column_width, item in zip(column_widths, row)
            ]
            for row in str_matrix
        ]

        markdown = [
            '| ' + ' | '.join(row) + ' |'
            for row in formatted_matrix
        ]

        markdown.insert(1, '|:' + ':|:'.join(['-' * column_width for column_width in column_widths]) + ':|')

        return '\n'.join(markdown)

    def absolute_to_md(self):
        """Constructs Markdown representation of confusion matrix

        Returns
        -------
        str
            string in Markdown representing confusion matrix

        """
        return self._matrix_to_md(self.absolute, name='gold\\predicted', formatting='d')

    def gold_wrt_predicted_to_md(self):
        """Constructs Markdown representation of matrix of P(gold|prediction)

        Returns
        -------
        str
            string in Markdown representing matrix of P(gold|prediction)

        """
        return self._matrix_to_md(100 * self.gold_wrt_predicted.transpose(), name='predicted\\gold', formatting='.2f')

    def predicted_wrt_gold_to_md(self):
        """Constructs Markdown representation of matrix of P(prediction|gold)

        Returns
        -------
        str
            string in Markdown representing matrix of P(prediction|gold)

        """
        return self._matrix_to_md(100 * self.predicted_wrt_gold, name='gold\\predicted', formatting='.2f')
