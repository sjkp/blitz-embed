FROM mcr.microsoft.com/devcontainers/cpp:1-debian-11

RUN apt-get update && apt-get install -y \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
    python3-pip \
    tar \
    gzip \
    make \
    wget

# Install GCC 10
RUN apt-get install -y gcc-10 g++-10

# Activate GCC 10 environment for subsequent commands
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10

# Verify the GCC version
RUN gcc --version && g++ --version
RUN curl https://cmake.org/files/v3.22/cmake-3.22.0-linux-x86_64.tar.gz --output /tmp/cmake.tar.gz
RUN mkdir /opt/cmake && \
    tar -xzvf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1
ENV PATH="/opt/cmake/bin:${PATH}"
RUN cmake --version

RUN apt-get install -y binutils

## Install model 
RUN curl -L -o /opt/bge-base-en-v1.5-q4_0.gguf https://huggingface.co/prithivida/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_0.gguf

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
