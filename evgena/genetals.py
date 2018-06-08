from typing import Callable, Sequence
from abc import ABC, abstractmethod

import numpy as np

from .model import Model, TfModel
from .data_transformations import augment_images
from genetals.core import ObjectiveFncBase, OperatorBase, GeneticAlgorithm, Population


class ImageSampler(ABC):
    @property
    @abstractmethod
    def sample(self) -> np.ndarray:
        NotImplemented

    @abstractmethod
    def next(self) -> None:
        NotImplemented


class TrivialImageSampler(ImageSampler):
    @property
    def sample(self) -> np.ndarray:
        return self._source_images

    def __init__(self, source_images: np.ndarray):
        super(TrivialImageSampler, self).__init__()

        self._source_images = source_images

    def next(self) -> None:
        pass


class DecayImageSampler(ImageSampler):
    @property
    def sample(self) -> np.ndarray:
        return self._source_images[self._sample_index.source_i]

    def __init__(
        self, source_images: np.ndarray, shuffle: bool = True,
        sample_size: int = 64, decay: float = 0.9
    ):
        super(DecayImageSampler, self).__init__()

        if sample_size >= len(source_images):
            raise ValueError(
                'Invalid sample size {} for {} source_images'.format(
                    sample_size, len(source_images)
            ))

        self._source_images = source_images
        self._shuffle = shuffle
        self._sample_size = sample_size
        self._decay = decay

        self._source_pos = len(self._source_images)
        self._source_index = np.empty(shape=0, dtype=np.int32)
        self._sample_index = np.recarray(
            (self._sample_size,),
            dtype=[('source_i', np.int32), ('chance', np.float32)]
        )
        self._sample_index.chance = 0

        self.next()

    def next(self) -> None:
        self._sample_index.chance *= self._decay
        replace_mask = self._sample_index.chance < np.random.random(self._sample_size)

        replace_count = replace_mask.sum()
        if replace_count > 0:
            past_end_count = self._source_pos + replace_count - len(self._source_images)
            if past_end_count > 0:
                pre_end = self._source_index[self._source_pos:].copy()
                if self._shuffle:
                    self._source_index = np.random.permutation(len(self._source_images))
                else:
                    self._source_index = np.arange(len(self._source_images))
                post_end = self._source_index[:past_end_count]
                replacements = np.concatenate((pre_end, post_end))
                self._source_pos = past_end_count
            else:
                replacements = self._source_index[self._source_pos:self._source_pos + replace_count]
                self._source_pos += replace_count

            self._sample_index.source_i[replace_mask] = replacements
            self._sample_index.chance = 1


class Images2LabelObjectiveFnc(ObjectiveFncBase):
    def __init__(
        self, model: Model, target_label: int, image_sampler: ImageSampler,
        similarity_measure: Callable[[np.ndarray, np.ndarray], np.ndarray]
        ):
        super(Images2LabelObjectiveFnc, self).__init__()

        self._model = model
        self._target_label = target_label
        self._image_sampler = image_sampler
        self._metrics = similarity_measure

    def __call__(self, genes: np.ndarray) -> np.ndarray:
        # fetch samples
        images = self._image_sampler.sample

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


class ClipOperator(OperatorBase):
    def __init__(self, input_op: OperatorBase, v_min: float, v_max: float):
        super(ClipOperator, self).__init__(input_op)

        self._v_min = v_min
        self._v_max = v_max

    def _operation(self, ga: GeneticAlgorithm, input_pop: Population) -> Population:
        clipped_genes = np.clip(input_pop.genes, a_min=self._v_min, a_max=self._v_max)

        return Population(clipped_genes, ga)


# TODO annealed ClipOperator


class FGSMMutation(OperatorBase):
    def __init__(
        self, original_op: OperatorBase,
        model: TfModel, image_sampler: ImageSampler, target_label: int,
        steps: int = 1, step_size: float = 0.002, max_diff: float = 0.2,
        is_targeted: bool = True
    ):
        super(FGSMMutation, self).__init__(original_op)

        self._model = model
        self._image_sampler = image_sampler
        self._target_label = target_label
        self._steps = steps
        self._step_size = step_size
        self._max_diff = max_diff
        self._is_targeted = is_targeted

        self._grad_sign = -1 if is_targeted else 1

    def _operation(self, ga: GeneticAlgorithm, original_pop: Population) -> Population:
        noise = images_to_BHWC(original_pop.genes)
        images = self._image_sampler = image_sampler

        for s_i in self._steps:
            # augment images
            augmented_images = np.clip(augment_images(noise, images), 0, 1)
            augmented_images_batch_shaped = augmented_images.reshape(-1, *augmented_images.shape[2:])

            labels = [target_label] * len(augmented_images_batch_shaped)

            curr_grads_batch_shaped = model.gradients(augmented_images_batch_shaped, labels)

            curr_noise = self._grad_sign * self._step_size * np.sign(
                curr_grads_batch_shaped.reshape(*augmented_images.shape[:-1], -1).mean(axis=1)
            )

            noise = np.clip(
                noise + curr_noise,
                - max_diff, max_diff
            )

        return Population(noise.reshape(original_pop.genes.shape), ga)

