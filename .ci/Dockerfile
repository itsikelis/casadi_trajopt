FROM --platform=x86_64 ubuntu:22.04

ARG USERNAME="robot"
ARG HOME_DIR="/home/${USERNAME}"

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="Europe/Paris"

# Installation of Ubuntu packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    python3-dev \
    python-is-python3 \
    python3-pip \
    git \
    cmake \
    build-essential \
    unzip \
    wget \
    libx11-dev \
    python3-tk \
    tk-dev \
    g++ \
    gcc \
    x11-apps \
    libeigen3-dev \
    pybind11-dev \
    libblas-dev \
    liblapack-dev \
    libmetis-dev \
    gfortran \
    libboost-all-dev \
    libtbb-dev \
    software-properties-common \
    coinor-libipopt-dev && \
    apt-get clean

# # CASADI
# RUN git clone https://github.com/EnricoMingo/casadi.git
# RUN cd casadi && \
#     git checkout main && \
#     git submodule init && \
#     git submodule update && \
#     mkdir -p build && \
#     cd build && \
#     apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y swig libeigen3-dev libboost-dev && \
#     cmake -DCMAKE_BUILD_TYPE:STRING=Release -DWITH_PYTHON=ON -DWITH_PYTHON3=ON -DWITH_THREAD=ON -DWITH_IPOPT=ON  .. && \
#     make -j8 && \
#     make install

RUN pip install casadi

ADD libhsl ${HOME_DIR}/libhsl
ENV LD_LIBRARY_PATH /home/robot/libhsl:$LD_LIBRARY_PATH

# Install conda (Miniconda is a smaller version of Anaconda)
RUN apt-get update && \
    apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# Add conda to the PATH
ENV PATH /opt/conda/bin:$PATH

# Create a Conda environment and install Pinocchio
RUN conda create -n pinocchio_env && conda install pinocchio -c conda-forge

# Activate the environment and set the default command
SHELL ["conda", "run", "-n", "pinocchio_env", "/bin/bash", "-c"]

# Set environment variables to ensure Conda environment is activated by default
ENV CONDA_DEFAULT_ENV=pinocchio_env
ENV PATH /opt/conda/envs/pinocchio_env/bin:$PATH

# Start a bash shell
WORKDIR ${HOME_DIR}
CMD ["bash"]
