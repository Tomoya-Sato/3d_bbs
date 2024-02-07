FROM nvidia/cuda:11.6.1-devel-ubuntu20.04 AS dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
WORKDIR /root/tmp
RUN sed -i 's@archive.ubuntu.com@jp.archive.ubuntu.com@g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    libsuitesparse-dev \
    wget \
    unzip \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libpcl-dev \
    libyaml-cpp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CMake 3.27
WORKDIR /root/tmp
RUN wget https://github.com/Kitware/CMake/archive/refs/tags/v3.27.9.zip && \
    unzip v3.27.9.zip && \
    cd CMake-3.27.9 && \
    ./bootstrap && \
    make && \
    make install

# Install Eigen 3.4.0
WORKDIR /root/tmp
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
    unzip eigen-3.4.0.zip
WORKDIR /root/tmp/eigen-3.4.0/build
RUN cmake .. && make install

FROM dependencies AS build
WORKDIR /root/workspace
COPY . /root/workspace/3d_bbs/
RUN ls /root/workspace && cd /root/workspace/3d_bbs/ && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4

FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04 AS deploy
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
WORKDIR /root/workspace
RUN sed -i 's@archive.ubuntu.com@jp.archive.ubuntu.com@g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y \
    libsuitesparse-dev \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libpcl-dev \
    libyaml-cpp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build /root/workspace/3d_bbs/ /root/workspace/
CMD ["/bin/bash"]
