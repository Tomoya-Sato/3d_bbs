# Load base image
FROM ghcr.io/mapiv/map4_tensorrt:22.02-py3

# Add maintainer info
LABEL maintainer="TaeYoung Kim <tyoung96@yonsei.ac.kr>"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Set essential packages
RUN apt-get update && apt-get install -y \
    libatlas-base-dev libgoogle-glog-dev \
    libsuitesparse-dev libglew-dev wget unzip git \
    python3-pip tzdata build-essential libboost-all-dev libeigen3-dev \
                         libflann-dev libvtk7-dev libqhull-dev libusb-dev \
                         libgtest-dev git-core freeglut3-dev pkg-config \
                         libxmu-dev libxi-dev libusb-1.0-0-dev libyaml-cpp-dev

# Install CMake 3.27
WORKDIR /root/tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.5/cmake-3.18.5-Linux-x86_64.tar.gz && \
    tar xvf *.tar.gz && \
    mv cmake-3.18.5-Linux-x86_64 /opt && \
    ln -f -s /opt/cmake-3.18.5-Linux-x86_64/bin/* /usr/bin && \
    # This specific version of cmake needs to be copied to /usr/local/bin as well. 
    # Otherwise, gpu-base will break because it has an older version of cmake there
    ln -f -s /opt/cmake-3.18.5-Linux-x86_64/bin/* /usr/local/bin && \
    rm cmake-3.18.5-Linux-x86_64.tar.gz

# Install Eigen 3.4.0
# WORKDIR /root/tmp
# RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && \
#     unzip eigen-3.4.0.zip
# WORKDIR /root/tmp/eigen-3.4.0/build
# RUN cmake .. && make install

# Install PCL
RUN apt-get install -y libpcl-dev

WORKDIR /root/workspace

# Load ROS environment at each run
COPY ./entrypoint.sh /
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]
