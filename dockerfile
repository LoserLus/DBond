FROM docker.1ms.run/pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV GEOGRAPHIC_AREA=Asia

# 更新源 + 安装系统工具 + 清理缓存
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list \
    && chmod 777 /tmp \
    && apt-get update  \
    && apt-get install -y sudo vim \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip（避免旧版本问题）
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

RUN pip install --upgrade pip

# 安装 Python 依赖
RUN pip install scikit-learn pandas pyteomics matplotlib positional-encodings[pytorch] tensorboard pandarallel

WORKDIR /workspace
