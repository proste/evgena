import io
import json
import os
import zipfile

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from wide_resnet import wide_resnet
from dataset import Dataset
from nsga import nsgaII


def random_interval_mask(batch_size, max_len):
    random_interval = tf.sort(tf.random.uniform([batch_size, 2], 0, max_len, dtype=tf.int32), axis=-1) + [0, 1]
    return tf.math.logical_xor(
        tf.sequence_mask(random_interval[:, 0], max_len),
        tf.sequence_mask(random_interval[:, 1], max_len)
    )


def orient_sequences(sequences, axis, max_dim):
    batch_size = sequences.get_shape()[0]
    return tf.reshape(sequences, shape=(batch_size, *([1] * axis), -1, *([1] * (max_dim - axis - 1))))


def two_point_xover(population, axes=None):
    pop_size, *example_shape = population.get_shape().as_list()
    assert not (pop_size % 2)
    
    pairs = tf.reshape(population, [2, -1, *example_shape])
        
    axes = axes or [i for i, _ in enumerate(example_shape)]
    xover_pattern = None
    for axis in axes:
        axis_mask = orient_sequences(random_interval_mask(pop_size // 2, example_shape[axis]), axis, len(example_shape))
        
        if xover_pattern is None:
            xover_pattern = axis_mask
        else:
            xover_pattern = tf.math.logical_and(xover_pattern, axis_mask)
    
    positive = tf.cast(xover_pattern, dtype=tf.float32)
    negative = tf.cast(tf.logical_not(xover_pattern), dtype=tf.float32)
    
    return tf.concat(
        (
            pairs[0] * positive + pairs[1] * negative,
            pairs[0] * negative + pairs[1] * positive,
        ),
        axis=0
    )


def mutation(population):
    perturbation = tf.random.normal(population.get_shape(), stddev=(4 / 255))
    return population + perturbation


def rgb_to_gray(images):
    return tf.expand_dims(0.3 * images[..., 0] + 0.59 * images[..., 1] + 0.11 * images[..., 2], axis=-1)


def choose_best_candidate(predictions, scores, intensities, target_label, is_targeted):
    labels = np.argmax(predictions, axis=-1)
    feasible_solutions = np.flatnonzero((labels == target_label) if is_targeted else (labels != target_label))

    if len(feasible_solutions):
        best_candidate_idx = feasible_solutions[intensities[feasible_solutions].argmin()]
    else:
        best_candidate_idx = scores.argmax()
    
    return best_candidate_idx


class Evgena:
    def __init__(self, model, pop_size, example_shape=(32, 32, 3)):
        self.pop_size = pop_size
        self.example_shape = example_shape
        
        self._session = tf.keras.backend.get_session()
        self.model = model
        self.target_image = tf.placeholder(tf.float32, shape=example_shape)
        self.population = tf.get_variable('population', shape=(self.pop_size, *self.example_shape), dtype=tf.float32, initializer=tf.initializers.zeros)
        
        shuffled = tf.random.shuffle(self.population)
        xovered = two_point_xover(shuffled, axes=[0, 1])
        mutated = mutation(xovered)
        # optional clip to perturbation norm
        self.perturbations = mutated
        
        target_image = tf.expand_dims(self.target_image, 0)
        perturbed = tf.clip_by_value(self.perturbations + target_image, 0, 1)
        quantized = tf.math.floor(256 * perturbed) / 256

        self.perturbation_intensities = 1 - tf.image.ssim(rgb_to_gray(quantized), rgb_to_gray(target_image), 1)
        self.predictions = model(quantized)
        
        self.to_replace = tf.placeholder(tf.int32, shape=[None])
        self.replacements = tf.placeholder(tf.float32, shape=[None, *self.example_shape])
        self.selection = tf.scatter_update(self.population, self.to_replace, self.replacements)
        
    def run(self, epochs, target_image, target_label, is_targeted):
        self._session.run(self.population.initializer)
        
        best_candidates = np.empty(
            shape=epochs,
            dtype=[
                ('image', np.float32, self.example_shape),
                ('predictions', np.float32, self.model.output_shape[1:]),
                ('intensity', np.float32)
            ]
        ).view(np.recarray)
        
        last_intensities, last_predictions = self._session.run(
            (self.perturbation_intensities, self.predictions),
            feed_dict={
                self.target_image: target_image,
                self.perturbations: self._session.run(self.population)
            }
        )
        for epoch_i in tqdm(range(epochs)):
            perturbations, intensities, predictions = self._session.run(
                (self.perturbations, self.perturbation_intensities, self.predictions),
                feed_dict={self.target_image: target_image}
            )
            
            all_predictions = np.concatenate((last_predictions, predictions))
            all_scores = (1 if is_targeted else -1) * all_predictions[:, target_label]
            all_intensities = np.concatenate((last_intensities, intensities))
            choice = nsgaII(np.stack([all_scores, - all_intensities], axis=-1), self.pop_size)
            
            replace_mask = np.zeros(shape=(2 * self.pop_size), dtype=np.bool)
            replace_mask[choice] = True
            
            to_replace = np.flatnonzero(~replace_mask[:self.pop_size])
            replacements = perturbations[replace_mask[self.pop_size:]]
            
            # update population
            self._session.run(
                self.selection,
                feed_dict={
                    self.to_replace: to_replace,
                    self.replacements: replacements
                }
            )
            
            # choose best candidate
            best_candidate_idx = choose_best_candidate(all_predictions, all_scores, all_intensities, target_label, is_targeted)
            if best_candidate_idx < self.pop_size:
                best_candidates[epoch_i] = best_candidates[epoch_i - 1]
            else:
                best_candidates[epoch_i] = (
                    np.clip(perturbations[best_candidate_idx - self.pop_size] + target_image, 0, 1),
                    predictions[best_candidate_idx - self.pop_size],
                    intensities[best_candidate_idx - self.pop_size]
                )
                
            last_intensities[to_replace] = intensities[replace_mask[self.pop_size:]]
            last_predictions[to_replace] = predictions[replace_mask[self.pop_size:]]
            
        return best_candidates
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=int)
    args = parser.parse_args()
    
    target = None if (args.target == -1) else args.target
    
    experiment_name = 'paper_preprocessing_dropout_decay'

    ds = Dataset.from_nprecord('datasets/cifar_10.npz')
    model = wide_resnet(28, 10)
    model.load_weights(experiment_name + '/weights-3.h5')

    evgena = Evgena(model, 256, example_shape=(32, 32, 3))

    dirname = f'target_{target}'
    arcname = f'{experiment_name}/{dirname}.zip'

    with zipfile.ZipFile(arcname, 'a') as zip_f:
        namelist = zip_f.namelist()
    
    for ex_i, ex in enumerate(tqdm(ds.test)):
        if ex.y == target:
            continue

        path = f'{dirname}/{ex_i}.npy'

        if path in namelist:
            continue

        if target is None:
            best_candidates = evgena.run(256, ex.X, ex.y, False)
        else:
            best_candidates = evgena.run(256, ex.X, target, True)
            
        with zipfile.ZipFile(arcname, 'a') as zip_f:
            with io.BytesIO() as temp_f:
                np.save(temp_f, best_candidates)
                zip_f.writestr(path, temp_f.getvalue(), compress_type=zipfile.ZIP_BZIP2)
        