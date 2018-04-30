from typing import Callable, Sequence

import numpy as np

from .models import Model
from .data_transformations import augment_images
from genetals.core import ObjectiveFncBase, InitializerBase


class Images2LabelObjectiveFnc(ObjectiveFncBase):
    def __init__(
        self, model: Model, similarity_measure: Callable[[np.ndarray, np.ndarray], np.ndarray],
        target_label: int, source_images: np.ndarray, shuffle: bool = True,
        sample_size: int = 64, sample_ttl: float = 0.9):
        super(Images2LabelObjectiveFnc, self).__init__()
        
        self._metrics = similarity_measure
        self._model = model
        self._target_label = target_label
        self._source_images = source_images
        self._sample_size = sample_size
        self._sample_ttl = sample_ttl
        self._shuffle_source = shuffle
        
        if self._shuffle_source:
            self._source_index = np.random.permutation(len(self._source_images))
        else:
            self._source_index = np.arange(len(self._source_images))
        
        self._samples = np.recarray((self._sample_size,), dtype=[('index', np.int32), ('ttl', np.float32)])
        self._samples.index = np.arange(self._sample_size)
        self._samples.ttl = 1
        
        self._source_i = self._sample_size
      
    def __call__(self, genes: np.ndarray) -> np.ndarray:
        # fetch samples
        images = self._source_images[self._source_index[self._samples.index]]
        
        # resolve ttl of samples
        self._samples.ttl *= self._sample_ttl
        death_mask = self._samples.ttl < np.random.random(len(self._samples))
        
        u_source_i = self._source_i + np.sum(death_mask)
        if  u_source_i > len(self._source_images):
            u_source_i -= len(self._source_images)
            babies = np.concatenate((np.arange(self._source_i, len(self._source_images)), np.arange(u_source_i)))
            np.random.shuffle(self._source_index)
        else:
            babies = np.arange(self._source_i, u_source_i)
        self._source_i = u_source_i
        
        self._samples.index[death_mask] = babies
        self._samples.ttl[death_mask] = 1
        
        # augment images
        augmented_images = augment_images(genes, images)
        augmented_images_batch_shaped = augmented_images.reshape(-1, *augmented_images.shape[2:])
        
        # for each individual sample its predictions, copmute ssim mean ssim
        norms = self._metrics(augmented_images_batch_shaped, np.expand_dims(images, 0).repeat(len(genes), axis=0).reshape(augmented_images_batch_shaped.shape))
        norms = norms.reshape(augmented_images.shape[:2])
        logits = self._model(augmented_images_batch_shaped)[:, self._target_label]
        logits = logits.reshape(augmented_images.shape[:2])
                       
        avg_norms = np.average(norms, axis=-1)
        avg_logits = np.average(logits, axis=-1)
        
        # create array by merging columns
        return np.stack((avg_logits, avg_norms), axis=-1)
