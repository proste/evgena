import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, SpatialDropout2D


def bn_relu_conv(x, filters, strides, dropout, weight_regularizer, name):
    x = BatchNormalization(scale=False, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    if dropout:
        x = SpatialDropout2D(dropout, name=name + '_do')(x)
    x = Conv2D(
        filters, 3, strides=strides, use_bias=False, padding='same',
        kernel_regularizer=weight_regularizer, name=name
    )(x)
    return x


def wide_resnet(depth, k, input_shape=(32, 32, 3), n_classes=10, dropout=None, weight_decay=None):
    n = (depth - 4) // 6
    assert (6 * n + 4) == depth

    weight_regularizer = tf.keras.regularizers.L1L2(l2=(weight_decay / 2)) if weight_decay else None
    
    x = pic = Input(shape=input_shape, name='pic')
    x = BatchNormalization(center=False, scale=False)(x)

    # conv1
    x = Conv2D(
        16, 3, use_bias=False, padding='same',
        kernel_regularizer=weight_regularizer, name='conv1'
    )(x)

    # convX
    for block_i in range(2, 5):
        filters = (2 ** block_i) * 4 * k
        for i in range(n):
            # whether to perform downsampling
            strides = (2, 2) if ((block_i > 2) and (not i)) else (1, 1)

            if i:
                shortcut = x
            else:
                shortcut = Conv2D(
                    filters, 1, strides=strides, use_bias=False, padding='same',
                    kernel_regularizer=weight_regularizer, name='proj{}'.format(block_i)
                )(x)

            x = bn_relu_conv(x, filters, strides, None, weight_regularizer, name=f'conv{block_i}a_{i}')
            x = bn_relu_conv(x, filters, (1, 1), dropout, weight_regularizer, name=f'conv{block_i}b_{i}')

            x = Add(name=f'conv{block_i}_{i}_merge')([shortcut, x])
    
    x = BatchNormalization(scale=False, name='head_bn')(x)
    x = Activation('relu', name='head_relu')(x)
    x = AveragePooling2D((8, 8), name='head_pool')(x)
    x = Flatten(name='head_flatten')(x)
    x = Dense(
        n_classes, activation='softmax', kernel_regularizer=weight_regularizer, name='head_out'
    )(x)

    return tf.keras.Model(inputs=pic, outputs=x)
