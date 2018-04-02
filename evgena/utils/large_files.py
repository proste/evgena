import os
import gzip
import urllib
import hashlib
from .file_system import copyfileobj


# lfs = {
#     "datasets": {
#         "emnist_balanced": {
#             "train_X": [],
#             "train_y": [],
#             "test_X": [],
#             "test_y": []
#         },
#         "mnist": {
#             "train_X": [],
#             "train_y": [],
#             "test_X": [],
#             "test_y": []
#         }
#     },
#     "models": {
#         "best_residual_dropout_nn_emnist_2.h5": []
#     }
# }

lfs = [
    (
        'datasets/emnist_balanced_train.npy',
        '5d6b4ee552c50687b56a92ce443438f612df06263c644596b1fad68451b72d2f',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/emnist_balanced_train.npy.gz'
    ), (
        'datasets/emnist_balanced_test.npy',
        'b0eb652b7bcd220b4382cdbde2132347e3ad4f3ada973032f74595fc56850255',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/emnist_balanced_test.npy.gz'
    ), (
        'datasets/mnist_train.npy',
        'e76d03149d49d074a0e7da5f5ccb2794ef3c92c518fd2a4238518fc355ad5b0f',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/mnist_train.npy.gz'
    ), (
        'datasets/mnist_test.npy',
        '86678d169b399e237d6d70bdda3c2406dcaff88b26f43047f84e1852b06dbe6f',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/mnist_test.npy.gz'
    ), (
        'models/best_residual_dropout_nn_emnist_2.h5',
        'cd218391e561e13391cabb290d9d7f9bafb9a1bc843c198c1e8a563179454f15',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/best_residual_dropout_nn_emnist_2.h5.gz'
    ), (
        'datasets/fashion_mnist.npz',
        'f6ed33b2819f8aaa4347c79c2909b60a57804a3b2048a4561fbdb73fb5632605',
        'http://www.ms.mff.cuni.cz/~prochas7/evgena/fashion_mnist.npz.gz'
    )
]


def file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Computes sha256 hash of file

    :param file_path: path to file
    :param chunk_size: size in bytes to be read at a time, -1 for reading whole contents
    :return: sha256 hash as a hex string
    """
    assert chunk_size != 0,\
        'chunk_size must not equal 0'

    checksum = hashlib.sha256()

    with open(file_path, 'rb') as file:
        chunk = file.read(chunk_size)
        while len(chunk) != 0:
            checksum.update(chunk)
            chunk = file.read(chunk_size)

    return checksum.hexdigest()


def maybe_download(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)

    for rel_path, checksum, url in lfs:
        if abs_path.endswith(rel_path):  # path recognized as large file
            if (not os.path.isfile(abs_path)) or (file_sha256(abs_path) != checksum):  # needs downloading
                os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                with urllib.request.urlopen(url) as response:
                    with gzip.GzipFile(fileobj=response) as in_file:
                        with open(file_path, 'wb') as out_file:
                            copyfileobj(in_file, out_file)

                if file_sha256(abs_path) != checksum:
                    raise ValueError('Downloaded large file {!r} checksum mismatch.')

            return file_path

    raise FileNotFoundError('{!r} is not recognized large file'.format(file_path))


# TODO how to get root dir of package??, force usage from root