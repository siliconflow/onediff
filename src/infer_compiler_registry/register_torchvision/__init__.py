from onediff.infer_compiler import register
import torchvision
import oneflow

torch2of_class_map = {
  torchvision.ops.misc.Conv2dNormActivation: oneflow.nn.Sequential,
  torchvision.ops.Conv2dNormActivation: oneflow.nn.Sequential
}

register(torch2oflow_class_map=torch2of_class_map)