import torch 
import oneflow as flow

class FakeCuda:
    @staticmethod
    def current_device():
        return "cuda:0"
    
    @staticmethod
    def mem_get_info(dev):
        return 1024*1024*1024 , 1024*1024*1024
    

flow.cuda.current_device = FakeCuda.current_device
flow.cuda.mem_get_info = FakeCuda.mem_get_info


