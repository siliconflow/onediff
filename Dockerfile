ARG BASE_IMAGE
ARG ONEFLOW_DEVICE
FROM ${BASE_IMAGE}

ARG ONEFLOW_PIP_INDEX
ARG ONEFLOW_PACKAGE_NAME=oneflow
RUN pip install -f ${ONEFLOW_PIP_INDEX} ${ONEFLOW_PACKAGE_NAME}
ADD . /src/onediff
RUN python3 -m pip install "torch" "transformers==4.27.1" "diffusers[torch]==0.15.1" && \
    python3 -m pip uninstall accelerate -y && \
    python3 -m pip install -e /src/onediff
