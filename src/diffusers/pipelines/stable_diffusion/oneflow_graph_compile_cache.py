import oneflow

from collections import deque
from timeit import default_timer as timer

global_config_cache_size = 1
global_cache_bucket = dict()


class OneFlowGraph(object):
    def __init__(self, graph_class, *args, **kwargs):
        self.graph_ = graph_class(*args, **kwargs)
        self.is_compiled_ = False

    @property
    def is_compiled(self):
        return self.is_compiled_

    def compile(self, *args, **kwargs):
        if self.is_compiled_:
            return

        global_class_name = self.graph_.__class__.__name__
        print(
            "[oneflow]",
            f"compiling {global_class_name} beforehand to make sure the progress bar is more accurate",
        )
        compilation_start = timer()
        compilation_time = 0
        self.graph_._compile(*args, **kwargs)
        compilation_time = timer() - compilation_start
        print("[oneflow]", "[elapsed(s)]", f"[{global_class_name} compilation]", f"{compilation_time:.3f}")

        self.is_compiled_ = True

    def __call__(self, *args, **kwargs):
        if not self.is_compiled_:
            self.compile(*args, **kwargs)

        return self.graph_(*args, **kwargs)


class LRUCache(object):
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = deque()
        self.hash_map = dict()

    def is_queue_full(self):
        return len(self.queue) == self.cache_size

    def set(self, key, value):
        if key in self.hash_map:
            return None

        if self.is_queue_full():
            pop_key = self.queue.pop()
            self.hash_map.pop(pop_key)
            self.queue.appendleft(key)
            self.hash_map[key] = value
            return pop_key
        else:
            self.queue.appendleft(key)
            self.hash_map[key] = value
            return key

    def get(self, key):
        if key in self.hash_map:
            self.queue.remove(key)
            self.queue.appendleft(key)
            return self.hash_map[key]

        return None


def set_oneflow_graph_compile_cache_size(cache_size):
    global global_config_cache_size
    global_config_cache_size = cache_size


def get_oneflow_graph_from_compile_cache(graph_class, cache_key, *args, **kwargs):
    global global_cache_bucket

    if graph_class.__name__ not in global_cache_bucket:
        global_cache_bucket[graph_class.__name__] = LRUCache(global_config_cache_size)

    compile_cache = global_cache_bucket[graph_class.__name__]

    graph = compile_cache.get(cache_key)
    if graph is None:
        graph = OneFlowGraph(graph_class, *args, **kwargs)
        ret = compile_cache.set(cache_key, graph)
        assert ret is not None

        if ret != cache_key:
            print(
                "[oneflow]",
                f"a {graph_class.__name__} with cache key {cache_key} "
                "is deleted from cache according to the LRU policy",
            )
            print("[oneflow]", "cache size can be changed by `set_oneflow_graph_compile_cache_size`")

        print("[oneflow]", f"a {graph_class.__name__} with cache key {cache_key} is appending to cache")

    return graph
