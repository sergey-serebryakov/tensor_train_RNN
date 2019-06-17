FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        python-setuptools \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

RUN pip --no-cache-dir install \
        tensorflow-gpu==1.13.1 \
        matplotlib \
        six
