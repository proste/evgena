import tensorflow as tf

from wide_resnet import wide_resnet
from dataset import Dataset
from large_files import maybe_download


def training_dataset(split, batch_size=128, padding=4):
    steps_per_epoch = len(split) // batch_size

    def flip_pad_crop(example):
        X = example['X']

        X = tf.image.random_flip_left_right(X)
        X = tf.pad(X, [(4, 4), (4, 4), (0, 0)], mode='REFLECT')
        X = tf.image.random_crop(X, (32, 32, 3))

        return (X, example['y'])

    tf_ds = tf.data.Dataset.from_tensor_slices({'X': split.X, 'y': split.y})
    tf_ds = tf_ds.repeat()
    tf_ds = tf_ds.shuffle(8 * batch_size)
    tf_ds = tf_ds.map(flip_pad_crop, num_parallel_calls=2)
    tf_ds = tf_ds.batch(batch_size, drop_remainder=False)
    tf_ds = tf_ds.prefetch(1)
    
    return steps_per_epoch, tf_ds


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('depth', type=int, default=28)
    parser.add_argument('k', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--tag')
    args = parser.parse_args

    if args.tag is None:
        args.tag = f'wide_res_net(depth={args.depth},k={args.k},dropout={args.dropout},weight_decay={args.weight_decay})'
    
    os.makedirs(args.tag)

    ds = Dataset.from_nprecord(maybe_download('datasets/cifar_10.npz'))
    model = wide_resnet(args.depth, args.k, dropout=args.dropout, weight_decay=args.weight_decay)
    
    steps_per_epoch, train_ds = training_dataset(ds.train)
    initial_epoch = 0
    for stage_i, epochs in enumerate([60, 60, 40, 40]):
        target_epoch = initial_epoch + epochs

        optimizer = tf.keras.optimizers.SGD(learning_rate=(0.1 * (0.2 ** stage_i)), momentum=0.9, nesterov=True)
        model.compile(optimizer, 'sparse_categorical_crossentropy', metrics=['acc'])

        history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=target_epoch, initial_epoch=initial_epoch)
        model.evaluate(ds.test.X, ds.test.y, batch_size=512)

        model.save('{}/weights-{}.h5'.format(args.tag, stage_i))
        with open('{}/history-{}.json'.format(args.tag, stage_i), 'w') as hist_f:
            json.dump(
                {k: [e.tolist() for e in v] for k, v in history.history.items()},
                hist_f,
                indent=2
            )

        initial_epoch = target_epoch
