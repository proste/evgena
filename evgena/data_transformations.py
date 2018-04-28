import numpy as np
import tensorflow as tf


class _ImageAugmentation:
    def __init__(self):
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)
        
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
                self.base_images + tf.expand_dims(resized_augmentations, 1), 0.0, 1.1
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

def images_to_BHWC(examples: np.ndarray, input_format: str = None) -> np.ndarray:
    if (input_format is not None) and (len(input_format) != examples.ndim):
        raise ValueError("input_format has different length from examples.ndim")
    
    if input_format == 'HW':
        return examples.reshape(1, *examples.shape, 1)
    elif input_format == 'BHW':
        return examples.reshape(*examples.shape, 1)
    elif input_format == 'HWC':
        return exampels.reshape(1, *example.shape)
    elif input_format is None:
        if examples.ndim == 2:                  # single gray image
            return examples.reshape(1, *examples.shape, 1)
        elif examples.ndim == 3:
            if examples.shape[2] in [1, 3, 4]:  # hopefully single gray, RGB, RGBA image
                return examples.reshape(1, *examples.shape)
            else:                               # multiple gray images
                return examples.reshape(*examples.shape, 1)
        elif examples.ndim == 4:                # already 4D BHWC
            return examples
        else:
            raise ValueError("Invalid shape of examples")
