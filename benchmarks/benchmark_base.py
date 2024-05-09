class BaseBenchmark:
    def __init__(self):
        pass

    def load_pipeline_from_diffusers(self, *args, **kwargs):
        raise NotImplementedError

    def compile_pipeline(self, *args, **kwargs):
        raise NotImplementedError

    def benchmark_model(self, *args, **kwargs):
        raise NotImplementedError
