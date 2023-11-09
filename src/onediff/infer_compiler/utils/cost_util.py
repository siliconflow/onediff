import oneflow as flow
import time


def cost_cnt(fn):
    def new_fn(*args, **kwargs):
        print("==> function ", fn.__name__, " try to run...")
        flow._oneflow_internal.eager.Sync()
        before_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print(fn.__name__, " cuda mem before ", before_used, " MB")
        before_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
        print(fn.__name__, " host mem before ", before_host_used, " MB")
        start_time = time.time()
        out = fn(*args, **kwargs)
        flow._oneflow_internal.eager.Sync()
        end_time = time.time()
        print(fn.__name__, " run time ", end_time - start_time, " seconds")
        after_used = flow._oneflow_internal.GetCUDAMemoryUsed()
        print(fn.__name__, " cuda mem after ", after_used, " MB")
        print(fn.__name__, " cuda mem diff ", after_used - before_used, " MB")
        after_host_used = flow._oneflow_internal.GetCPUMemoryUsed()
        print(fn.__name__, " host mem after ", after_host_used, " MB")
        print(fn.__name__, " host mem diff ", after_host_used - before_host_used, " MB")
        print("<== function ", fn.__name__, " finish run.")
        print("")
        return out

    return new_fn
