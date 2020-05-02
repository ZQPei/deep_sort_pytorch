from functools import wraps
from time import time

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def is_video(ext: str):
    """
    Check if a video input is in supported foramts.
    Args:
        ext:

    Returns:

    """
    allowed_exts = ('.mp4', '.webm', '.ogg', '.avi')
    return any((ext.endswith(x) for x in allowed_exts))


def tik_tok(func):
    """
    keep track of time for each process. editted for decorator usages
    Args:
        func:

    Returns:

    """
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time()
            print("time: {:.03f}s, fps: {:.03f}".format(end_ - start, 1 / (end_ - start)))

    return _time_it

