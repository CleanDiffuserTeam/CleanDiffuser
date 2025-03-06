#########################################################   
# Dockerfile for CleanDiffuser                          #
# Yifu Yuan, 2025 (c)                                   #
# ------------------------------------------------------#
# Build instructions:                                   #
# docker build . -t cleandiffuser:1.1.0                 #
# docker push cleandiffuser:1.1.0                       #
# ------------------------------------------------------#
# Run:                                                  #
# docker run -i \                                       #
#   -v <path>/<to>/cleandiffuser:/cleandiffuser \       #
#   --gpus all \                                        #
#   -t cleandiffuser:1.1.0 \                            #
#   /bin/bash                                           #
#########################################################
# Everything tests well in V100 GPU

# base image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

# Install initial packages and OpenMPI dependencies
RUN sed -i 's@/archive.ubuntu.com/@/mirrors.cloud.tencent.com/@g' /etc/apt/sources.list && \
    apt-get -y update && \
    apt-get install -y --no-install-recommends build-essential git vim tree curl wget \
    ffmpeg unzip htop xvfb ca-certificates bash-completion libjpeg-dev libpng-dev \
    libssl-dev libcurl4-openssl-dev libopenmpi-dev zlib1g-dev \
    libglib2.0-0 libglu1-mesa-dev libgl1-mesa-dev libvulkan1 libgl1-mesa-glx libosmesa6 \
    libosmesa6-dev libglew-dev mesa-utils libfuse2 libgoogle-perftools4 libnspr4 libnss3 \
    fuse libibverbs1 openssh-server openssh-client iputils-ping iproute2 telnet expect && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /root/.ssh /run/sshd

# Use Tsinghua mirror for Conda
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r/ && \
    conda config --set show_channel_urls yes && \
    pip config set global.index-url https://mirrors.tencent.com/pypi/simple/ && \
    pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple && \
    pip config set global.trusted-host mirrors.tencent.com

# conda environment
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
SHELL ["/bin/bash", "-c"]
RUN conda init && \
    echo "cd /root" >> /root/.bashrc

# cleandiffuser
RUN pip install git+https://github.com/CleanDiffuserTeam/CleanDiffuser && \
    pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl && \
    pip install lightning && \
    pip install setuptools==65.5.0 pip==21 && \
    pip install gym==0.21.0 mujoco_py>=2.0

# mujoco 2.1.0
ENV MUJOCO_GL egl
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
RUN mkdir -p /root/.mujoco && \
    wget https://www.tdmpc2.com/files/mjkey.txt && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz && \
    mv mujoco210 /root/.mujoco/mujoco210 && \
    mv mjkey.txt /root/.mujoco/mjkey.txt && \
    find /root/.mujoco -uid 421709 -exec chown root:root {} \; && \
    python -c "import mujoco_py"

# metaworld
RUN pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb

# Install OpenMPI
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz && \
    tar xzvf openmpi-4.0.7.tar.gz && \
    cd openmpi-4.0.7 && \
    ./configure --prefix=/usr/local --enable-orterun-prefix-by-default && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -r openmpi-4.0.7*
    
# Set bash as default shell
RUN ln -sf /bin/bash /bin/sh

# Configure SSH
RUN mkdir -p ~/.ssh && \
    touch ~/.ssh/config && \
    touch /root/.ssh/authorized_keys && \
    echo "HostKey /etc/ssh/ssh_host_rsa_key" >> /etc/ssh/sshd_config && \
    echo "HostKey /etc/ssh/ssh_host_ecdsa_key" >> /etc/ssh/sshd_config && \
    echo "HostKey /etc/ssh/ssh_host_ed25519_key" >> /etc/ssh/sshd_config && \
    echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

# success!
RUN echo "Successfully built CleanDiffuser Docker image!"