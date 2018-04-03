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
            x_data, y_data = np.mgrid[-window_half:window_half+1,-window_half:window_half+1]

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
            sigma_x_sq = tf.nn.conv2d(self.images_x*self.images_x, window, strides=[1, 1, 1, 1], padding='VALID') - mu_x_sq
            sigma_x = tf.sqrt(sigma_x_sq)
            
            mu_y = tf.nn.conv2d(self.images_y, window, strides=[1, 1, 1, 1], padding='VALID')
            mu_y_sq = mu_y * mu_y
            sigma_y_sq = tf.nn.conv2d(self.images_y*self.images_y, window, strides=[1, 1, 1, 1], padding='VALID') - mu_y_sq
            sigma_y = tf.sqrt(sigma_y_sq)
            
            mu_xy = mu_x * mu_y
            sigma_xy = tf.nn.conv2d(self.images_x*self.images_y, window, strides=[1, 1, 1, 1], padding='VALID') - mu_xy
            
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