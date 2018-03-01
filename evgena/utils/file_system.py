import contextlib
from typing import Callable

__all__ = ['maybe_open', 'copyfileobj']


@contextlib.contextmanager
def maybe_open(path_or_fp, *args, **kwargs):
    if isinstance(path_or_fp, str):
        f = file_to_close = open(path_or_fp, *args, **kwargs)
    else:
        f = path_or_fp
        file_to_close = None

    try:
        yield f
    finally:
        if file_to_close:
            file_to_close.close()


def copyfileobj(fsrc, fdst, length: int = 65536, callback: Callable =None):
    copied = 0
    while True:
        buf = fsrc.read(length)

        if len(buf) == 0:
            break

        fdst.write(buf)
        copied += len(buf)

        if callback is not None:
            callback(copied)
