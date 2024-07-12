from abc import ABC, abstractmethod


class BaseBenchmark(ABC):
    @abstractmethod
    def load_pipeline_from_diffusers(self, *args, **kwargs):
        pass

    @abstractmethod
    def compile_pipeline(self, *args, **kwargs):
        pass

    @abstractmethod
    def benchmark_model(self, *args, **kwargs):
        pass
