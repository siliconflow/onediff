#==== Generated from ./docker/config/community-default.yaml ====
#     yaml file SHA256: d2822e8356b6a9480114a27cf761158a057f38074a1227e4e207c4050306569d


#==== Docker Base Image ====
FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

#==== Install the OneFlow ====
RUN pip install --pre oneflow -f https://oneflow-pro.oss-cn-beijing.aliyuncs.com/branch/main/cu122


#==== Download and set up the repos ====
COPY onediff /app/onediff
WORKDIR /app/onediff
RUN  python3 -m pip install transformers==4.27.1 diffusers[torch]==0.25.0 && \
python3 -m pip install -e . && \
cd onediff_diffusers_extensions && python3 -m pip install -e .


#==== Post setting
WORKDIR /app
