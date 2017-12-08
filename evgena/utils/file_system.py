import contextlib

__all__ = ['maybe_open']


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
