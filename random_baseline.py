import io
import json
import os
import zipfile

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from wide_resnet import wide_resnet
from dataset import Dataset


def rgb_to_gray(images):
    return tf.expand_dims(0.3 * images[..., 0] + 0.59 * images[..., 1] + 0.11 * images[..., 2], axis=-1)


class RandomBaseline:
    def __init__(self, model, batch_size=2048, example_shape=(32, 32, 3)):
        self._session = tf.keras.backend.get_session()
        
        self.batch_size = batch_size
        self.model = model
        self.example_shape = example_shape
        self.target_image = tf.placeholder(tf.float32, shape=example_shape)
        self._target_image = tf.expand_dims(self.target_image, 0)
        self.cache = {}
        
    def _build(self, stddev):
        perturbations = tf.random.normal(shape=((self.batch_size,) + self.example_shape), stddev=stddev)
        perturbed = tf.clip_by_value(perturbations + self._target_image, 0, 1)
        quantized = tf.math.floor(256 * perturbed) / 256
        intensities = 1 - tf.image.ssim(rgb_to_gray(perturbed), rgb_to_gray(self._target_image), 1)
        predictions = self.model(perturbed)
        
        return quantized, predictions, intensities
        
        
    def run(self, eval_count, target_image, target_label, is_targeted, stddev):
        if stddev not in self.cache:
            self.cache[stddev] = self._build(stddev)
        
        images_tf, predictions_tf, intensities_tf = self.cache[stddev]
        
        highest_score, lowest_intensity = -1, 1
        best_candidate = None
        
        intensities_acc = []
        for _ in tqdm(range(eval_count // self.batch_size)):
            images, predictions, intensities = self._session.run(
                (images_tf, predictions_tf, intensities_tf),
                feed_dict={self.target_image: target_image}
            )
            intensities_acc.append(intensities)
            
            predicted_labels = np.argmax(predictions, axis=-1)
            feasible_solutions = np.flatnonzero(np.logical_and(
                (predicted_labels == target_label) if is_targeted else (predicted_labels != target_label),
                predictions.max(axis=-1) > 0.5
            ))
            if len(feasible_solutions):
                best_candidate_idx = feasible_solutions[intensities[feasible_solutions].argmin()]
                intensity = intensities[best_candidate_idx]
                if intensity < lowest_intensity:
                    lowest_intensity = intensity
                    highest_score = (1 if is_targeted else -1) * predictions[best_candidate_idx, target_label]
                    best_candidate = (images[best_candidate_idx], predictions[best_candidate_idx], intensity)
                    
            else:
                scores = (1 if is_targeted else -1) * predictions[:, target_label]
                best_candidate_idx = scores.argmax()
                score = scores[best_candidate_idx]
                if score > highest_score:
                    highest_score = score
                    lowest_intensity = intensities[best_candidate_idx]
                    best_candidate = (images[best_candidate_idx], predictions[best_candidate_idx], lowest_intensity)

        return best_candidate


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
    
    eval_count = 256 * 256
    
    dirname = f'baseline_target_{target}'
    arcname = f'{experiment_name}/{dirname}.zip'
    
    with zipfile.ZipFile(arcname, 'a') as zip_f:
        namelist = zip_f.namelist()
        
    random_baseline = RandomBaseline(model)
    
    for ex_i, ex in enumerate(tqdm(ds.test)):
        if ex.y == target:
            continue
        
        path = f'{dirname}/{ex_i}.npy'

        if path in namelist:
            continue
        
        candidate, predictions, intensity = random_baseline.run(
            eval_count, ex.X, (ex.y if (target is None) else target), (target is not None), 0.03 
        )
        
        with zipfile.ZipFile(arcname, 'a') as zip_f:
            with io.BytesIO() as temp_f:
                np.savez(temp_f, intensity=intensity, predictions=predictions, image=candidate)
                zip_f.writestr(path, temp_f.getvalue(), compress_type=zipfile.ZIP_BZIP2)
