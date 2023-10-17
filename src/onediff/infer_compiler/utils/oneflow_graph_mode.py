_ONEFLOW_GRAPH_MODE = False


class oneflow_graph_mode(object):
    def __init__(self, enabled=None):
        if enabled is not None:
            self.enabled = enabled
        else:
            self.enabled = True

    def __enter__(self):
        global _ONEFLOW_GRAPH_MODE
        self.prev_mode = _ONEFLOW_GRAPH_MODE
        _ONEFLOW_GRAPH_MODE = self.enabled

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _ONEFLOW_GRAPH_MODE
        _ONEFLOW_GRAPH_MODE = self.prev_mode


def oneflow_graph_mode_enabled():
    global _ONEFLOW_GRAPH_MODE
    return _ONEFLOW_GRAPH_MODE
