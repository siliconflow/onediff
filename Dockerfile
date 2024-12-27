ARG BASE_IMAGE
ARG ONEFLOW_DEVICE
FROM ${BASE_IMAGE}

ARG ONEFLOW_PIP_INDEX
ARG ONEFLOW_PACKAGE_NAME=oneflow
RUN pip install -U --pre -f ${ONEFLOW_PIP_INDEX} ${ONEFLOW_PACKAGE_NAME} "nvidia-cudnn-cu11>=8.9,<9.0"
RUN python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.19.3" "huggingface-hub==0.23.2" nvidia-cuda-cupti-cu12 nvidia-nvjitlink-cu12
ADD . /src/onediff
RUN python3 -m pip install -e /src/onediff
RUN python3 -m pip install -e /src/onediff/onediff_diffusers_extensions
