from typing import Sequence

import numpy as np
import tensorflow as tf


# TODO add typing
class _ImageAugmentation:
    def __init__(self):
        graph = tf.Graph()
        config=tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.session = tf.Session(
            graph=graph,
            config=config
        )

        with graph.as_default():
            # input placeholders
            self.augmentations = tf.placeholder(tf.float32, [None, None, None, 1], name='augmentations')
            self.base_images = tf.placeholder(tf.float32, [None, None, None, 1], name='base_images')  # TODO link dimensions??

            # resize augmentations to match images
            resized_augmentations = tf.image.resize_images(
                self.augmentations, tf.shape(self.base_images)[1:3],
                method=tf.image.ResizeMethod.BILINEAR, align_corners=True
            )

            # add together with augmentations reshaped
            self.augmented_images = tf.clip_by_value(
                self.base_images + tf.expand_dims(resized_augmentations, 1), 0.0, 1.0
            )

    def __call__(self, augmentations, base_images):
        augmentations = images_to_BHWC(augmentations)
        base_images = images_to_BHWC(base_images)

        return self.session.run(
            self.augmented_images,
            feed_dict={self.augmentations: augmentations, self.base_images: base_images}
        )


_image_augmentation = None
def augment_images(
    augmentations: np.ndarray, base_images: np.ndarray, batch_size: int = -1
) -> np.ndarray:
    """Additively augment images

    Parameters
    ----------
    augmentations : np.ndarray
        array of size XHWC in [0.0, 1.0], where X is number of augmentations
    base_images : np.ndarray
        array of size YHWC in [0.0, 1.0], where Y is number of images to be augmented
    batch_size : int
        take at most batch_size augmentations at a time; default -1 for no batching

    Returns
    -------
    np.ndarray
        array of size XYHWC in [0.0, 1.0], where [X, Y] is image Y augmented with X

    """
    global _image_augmentation
    if _image_augmentation is None:
        _image_augmentation = _ImageAugmentation()

    if batch_size < 0:  # no batching
        return _image_augmentation(augmentations, base_images)
    else:               # batching
        results = []
        for batch_begin in range(0, len(augmentations), batch_size):
            batch_end = batch_begin + batch_size
            results.append(_image_augmenation(
                augmentations[batch_begin:batch_end], base_images
            ))

        return np.concatenate(results)


def shape_to_BHWC(shape: Sequence[int], input_format: str = None) -> Sequence[int]:
    """Deduce BHWC normalized shape

    If auto-deduction used, the rules are following:
        [x, y] -> 'HW'
        [x, y, (1|3|4)] -> 'HWC'
        [x, y, ^(1|3|4)] -> 'BHW'
        [w, x, y, z] -> 'BHWC'

    Parameters
    ----------
    shape : Sequence[int]
        shape to be normalized
    input_format : {'HW', 'HWC', 'BHW', 'BHWC'}, optional
        format of shape to be normalized; defaults to None ie. auto-deduce

    Returns
    -------
    Sequence[int]
        BHWC normalized shape

    """
    if (input_format is not None) and (len(input_format) != len(shape)):
        raise ValueError('Cannot match input format with given shape')

    if input_format is None:
        if len(shape) == 2:            # single gray image
            input_format = 'HW'
        elif len(shape) == 3:
            if shape[2] in [1, 3, 4]:  # hopefully single gray, RGB, RGBA image
                input_format = 'HWC'
            else:                      # multiple gray images
                input_format = 'BHW'
        elif len(shape) == 4:          # already BHWC
            input_format = 'BHWC'
        else:
            raise ValueError('Cannot convert shape {!r}'.format(shape))

    if input_format == 'HW':
        return (1, *shape, 1)
    elif input_format == 'BHW':
        return (*shape, 1)
    elif input_format == 'HWC':
        return (1, *shape)
    elif input_format == 'BHWC':
        return shape
    else:
        raise ValueError('Cannot resolve {!r} input format'.format(input_format))


def images_to_BHWC(examples: np.ndarray, input_format: str = None) -> np.ndarray:
    """Reshapes examples to BHWC format

    Equivalent to examples.reshape(shape_to_BHWC(examples.shape, input_format)) call

    Parameters
    ----------
    examples : np.ndarray
        examples to be reshaped
    input_format : str
        see ``shape_to_BHWC`` for more info

    Returns
    -------
    np.ndarray
        view of ``examples`` with BHWC normalized shape

    """
    return examples.reshape(shape_to_BHWC(examples.shape, input_format))


def decode_labels(labels: np.ndarray, label_count: int) -> np.ndarray:
    """Decodes labels if needed

    Parameters
    ----------
    labels : np.ndarray
        array of possibly one-hot encoded labels
            - 1D for one-hot encoded labels
            - 2D for sparse labels - does nothing
    label_count : int
        number of labels to be used in one-hot decoding

    Returns
    -------
    np.ndarray
        decoded labels of shape len(labels) x label_count

    """
    if labels.ndim == 1:
        decoded = np.zeros(shape=(len(labels), label_count), dtype=np.float32)
        decoded[np.arange(len(labels)), labels] = 1
        return decoded
    elif labels.ndim == 2:
        return labels
    else:
        raise ValueError('Invalid labels shape: {}'.format(labels.shape))


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encodes labels if needed

    Parameters
    ----------
    labels : np.ndarray
        array of possibly sparse labels
            - 1D for one-hot encoded labels - does nothing
            - 2D for sparse labels

    Returns
    -------
    np.ndarray
        one-hot encoded labels

    """
    if labels.ndim == 1:
        return labels
    elif labels.ndim == 2:
        return np.argmax(labels, axis=-1)
    else:
        raise ValueError('Invalid labels shape: {}'.format(labels.shape))
