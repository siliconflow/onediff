import time
from contextlib import contextmanager


@contextmanager
def track_inference_time(warmup=False):
    """
    A context manager to measure the execution time of models.
    Parameters:
        warmup (bool): If True, prints the time for warmup runs; otherwise, prints the time for normal runs.
    """
    try:
        start_time = time.time()
        yield
    finally:
        end_time = time.time()
        if warmup:
            print(f"Warmup run - Execution time: {end_time - start_time:.2f} seconds")
        else:
            print(f"Normal run - Execution time: {end_time - start_time:.2f} seconds")
