from typing import Callable, IO


def copyfileobj(
    fsrc: IO, fdst: IO, length: int = 65536,
    callback[[int], None]: Callable = None
) -> None:
    """Copies data between file-like objects

    Parameters
    ----------
    fsrc : IO
        file-like source
    fdst : IO
        file-like destination
    length : int
        size of buffer to use during copy
    callback : Callable[[int], None]
        optional function called after each chunk is processed;
        number of already copied bytes is passed to the function

    """
    copied = 0
    while True:
        buf = fsrc.read(length)

        if len(buf) == 0:
            break

        fdst.write(buf)
        copied += len(buf)

        if callback is not None:
            callback(copied)
