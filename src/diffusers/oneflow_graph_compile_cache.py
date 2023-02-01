import os
from collections import deque
from timeit import default_timer as timer
from .utils import logging
import oneflow as flow

logger = logging.get_logger(__name__)


class OneFlowGraph(object):
    def __init__(self, graph_class, *args, **kwargs):
        self.graph_ = graph_class(*args, **kwargs)
        self.is_compiled_ = False
        self.is_shared_from_ = False

    @property
    def is_compiled(self):
        return self.is_compiled_

    def compile(self, *args, **kwargs):
        if self.is_compiled_:
            return

        global_class_name = self.graph_.__class__.__name__
        logger.info(
            f"[oneflow] compiling {global_class_name} beforehand to make sure the progress bar is more accurate",
        )
        compilation_start = timer()
        compilation_time = 0
        if self.is_shared_from_:
            self.graph_._compile_from_shared(*args, **kwargs)
        else:
            self.graph_._compile(*args, **kwargs)
        compilation_time = timer() - compilation_start
        logger.info(f"[oneflow] [elapsed(s)] [{global_class_name} compilation] {compilation_time:.3f}")

        self.is_compiled_ = True

    def load_runtime_state_dict(self, state_dict):
        if self.is_compiled_:
            return

        global_class_name = self.graph_.__class__.__name__
        logger.info(
            f"[oneflow] loading {global_class_name} beforehand to make sure the progress bar is more accurate",
        )
        load_start = timer()
        load_time = 0
        self.graph_.load_runtime_state_dict(state_dict)
        load_time = timer() - load_start
        logger.info(f"[oneflow] [elapsed(s)] [{global_class_name} loading] {load_time:.3f}")

        self.is_compiled_ = True

    def share_from(self, other_graph):
        self.graph_.share_from(other_graph.graph_)
        self.is_shared_from_ = True

    def __call__(self, *args, **kwargs):
        if not self.is_compiled_:
            self.compile(*args, **kwargs)

        return self.graph_(*args, **kwargs)


class LRUCache(object):
    _cnt: int = 0
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.queue = deque()
        self.hash_map = dict()

    def front(self):
        if self.is_empty():
            return None

        key = self.queue[0]
        return self.hash_map[key]

    def is_empty(self):
        return len(self.queue) == 0

    def is_queue_full(self):
        return len(self.queue) >= self.cache_size

    def pop(self):
        pop_key = self.queue.pop()
        value = self.hash_map.pop(pop_key)
        del value
        return pop_key

    def set(self, key, value):
        if key in self.hash_map:
            return None

        pop_key = None
        while self.is_queue_full():
            pop_key = self.pop()

        self.queue.appendleft(key)
        value._oneflow_graph_cache_order = LRUCache._cnt
        LRUCache._cnt += 1
        self.hash_map[key] = value
        return pop_key if pop_key is not None else key

    def get(self, key):
        if key in self.hash_map:
            self.queue.remove(key)
            self.queue.appendleft(key)
            return self.hash_map[key]

        return None
    
    def pairs(self):
        for (key, value) in self.hash_map.items():
            yield (key, value)



class OneFlowGraphCompileCache(object):
    def __init__(self, cache_size=1, enable_graph_share_mem=False):
        self.cache_size_ = cache_size
        self.cache_bucket_ = dict()
        self.share_origin_ = dict()
        self.enable_share_mem_ = enable_graph_share_mem
        self.enable_save_graph_ = False
        self.enable_load_graph_ = False
        self.graph_save_load_path_ = None

    def set_cache_size(self, cache_size):
        self.cache_size_ = cache_size

        for cache in self.cache_bucket_.values():
            cache.cache_size = cache_size

    def enable_share_mem(self, enabled=True):
        self.enable_share_mem_ = enabled
    
    def enable_save_graph(self, enabled=True):
        self.enable_save_graph_ = enabled

    def enable_load_graph(self, enabled=True):
        self.enable_load_graph_ = enabled

    def save_graph(self, path):
        if self.enable_save_graph_:
            for (graph_class_name, cache) in self.cache_bucket_.items():
                for (key, graph) in cache.pairs():
                    state_dict = graph.graph_.runtime_state_dict()
                    state_dict["cache_order"] = graph._oneflow_graph_cache_order
                    state_dict["cache_key"] = key
                    state_dict["graph_class_name"] = graph_class_name
                    flow.save(state_dict, os.path.join(path, graph_class_name + "_" + str(hash(key))))

    def load_graph(self, path, graph_class2init_args=None):
        if self.enable_load_graph_:
            sub_files = [ f.path for f in os.scandir(path) if f.is_file() ]
            graph_dict = dict()
            for sub_file in sub_files:
                state_dict = flow.load(sub_file)
                cache_order = state_dict["cache_order"]
                graph_dict[cache_order] = state_dict
            
            for order, state_dict in sorted(graph_dict.items()):
                graph_class_name  = state_dict["graph_class_name"]
                cache_key = state_dict["cache_key"]
                if graph_class_name not in self.cache_bucket_:
                    self.cache_bucket_[graph_class_name] = LRUCache(self.cache_size_)
                compile_cache = self.cache_bucket_[graph_class_name]
                if graph_class_name in graph_class2init_args:
                    init_args = graph_class2init_args[graph_class_name]
                    graph = OneFlowGraph(init_args[0], init_args[1])
                else:
                    graph = OneFlowGraph(flow.nn.Graph)
                if self.enable_share_mem_ is True:
                    if graph_class_name in self.share_origin_:
                        graph.share_from(self.share_origin_[graph_class_name])
                    else:
                        self.share_origin_[graph_class_name] = graph
                        graph.graph_.enable_shared()

                graph.load_runtime_state_dict(state_dict)
                ret = compile_cache.set(cache_key, graph)
                assert ret is not None

    def get_graph(self, graph_class, cache_key, *args, **kwargs):
        graph_class_name = graph_class.__name__
        if graph_class_name not in self.cache_bucket_:
            self.cache_bucket_[graph_class_name] = LRUCache(self.cache_size_)

        compile_cache = self.cache_bucket_[graph_class_name]
        graph = compile_cache.get(cache_key)

        if graph is None:
            graph = OneFlowGraph(graph_class, *args, **kwargs)
            ret = compile_cache.set(cache_key, graph)
            assert ret is not None
            if self.enable_save_graph_:
                graph.graph_.enable_save_runtime_state_dict()
            if self.enable_share_mem_ is True:
                if graph_class_name in self.share_origin_:
                    graph.share_from(self.share_origin_[graph_class_name])
                else:
                    self.share_origin_[graph_class_name] = graph
                    graph.graph_.enable_shared()

            if ret != cache_key:
                logger.info(
                    f"[oneflow] a {graph_class_name} with cache key {ret} "
                    "is deleted from cache according to the LRU policy"
                )
                if self.cache_size_ == 1:
                    logger.info("[oneflow] cache size can be changed by `set_cache_size`")

            logger.info(
                f"[oneflow] a {graph_class_name} with cache key {cache_key} is appended to "
                f"cache (cache_size={len(compile_cache.queue)}/{compile_cache.cache_size})"
            )

        return graph
