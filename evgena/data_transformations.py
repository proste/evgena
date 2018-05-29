from typing import Tuple

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
        # TODO re-create session every time?
        augmentations = images_to_BHWC(augmentations)        
        base_images = images_to_BHWC(base_images)
        
        return self.session.run(
            self.augmented_images,
            feed_dict={self.augmentations: augmentations, self.base_images: base_images}
        )

    
_image_augmentation = None
def augment_images(augmentations, base_images):
    global _image_augmentation
    if _image_augmentation is None:
        _image_augmentation = _ImageAugmentation()
    
    return _image_augmentation(augmentations, base_images)

def shape_to_BHWC(shape: Sequence[int], input_format: str = None) -> Sequence[int]:
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
        return (*shape)
    else:
        raise ValueError('Cannot resolve {!r} input format'.format(input_format))

def images_to_BHWC(examples: np.ndarray, input_format: str = None) -> np.ndarray:
    return examples.reshape(shape_to_BHWC(examples.shape, input_format))
