import oneflow as flow
import time
from .log_utils import LOGGER


def cost_cnt(debug=False):
    def decorate(func):
        def clocked(*args, **kwargs):
            if not debug:
                return func(*args, **kwargs)

            LOGGER.debug(f"==> function {func.__name__}  try to run...")
            flow._oneflow_internal.eager.Sync()

            before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            LOGGER.debug(f"{func.__name__} cuda mem before {before_used} MB")

            before_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
            LOGGER.debug(f"{func.__name__} host mem before {before_host_used} MB")

            start_time = time.time()
            out = func(*args, **kwargs)
            flow._oneflow_internal.eager.Sync()
            end_time = time.time()

            LOGGER.debug(f"{func.__name__} run time {end_time - start_time} seconds")

            after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
            LOGGER.debug(f"{func.__name__} cuda mem after {after_used} MB")

            LOGGER.debug(f"{func.__name__} cuda mem diff {after_used - before_used} MB")
            after_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
            LOGGER.debug(f"{func.__name__} host mem after {after_host_used} MB")
            LOGGER.debug(
                f"{func.__name__} host mem diff {after_host_used - before_host_used} MB"
            )

            LOGGER.debug(f"<== function {func.__name__} finish run.")
            LOGGER.debug("")
            return out

        return clocked

    return decorate
