ARG BASE_IMAGE
ARG ONEFLOW_DEVICE
FROM ${BASE_IMAGE}

ARG ONEFLOW_PIP_INDEX
ARG ONEFLOW_PACKAGE_NAME=oneflow
RUN python3 -m pip install "transformers==4.27.1" "diffusers[torch]==0.19.3" "huggingface_hub==0.25.0" ${ONEFLOW_PACKAGE_NAME} -f ${ONEFLOW_PIP_INDEX} && python3 -m pip cache purge
ADD . /src/onediff
RUN python3 -m pip install -e /src/onediff
RUN python3 -m pip install -e /src/onediff/onediff_diffusers_extensions
