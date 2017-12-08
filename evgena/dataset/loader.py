import os
from . import idx


def load_idx_dataset(dir_name):
    train_X, train_y, test_X, test_y = (
        idx.to_ndarray(os.path.join(dir_name, table_path))
        for table_path in ['train_X', 'train_y', 'test_X', 'test_y']
    )

    with open(os.path.join(dir_name, 'mapping'), 'r') as map_file:
        mapping = {
            int(key): [val for val in vals.split(',')]
            for key, vals in (
                line.rstrip('\n\r').split(':', 2)
                for line in map_file
            )
        }

    # TODO possibly add size checks (assert/exception)

    return (
        (train_X, train_y),
        (test_X, test_y),
        mapping
    )

