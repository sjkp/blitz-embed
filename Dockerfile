FROM almalinux:8 as build

# Install development tools and additional dependencies
RUN dnf install -y \
    git \
    libcurl-devel \
    openssl-devel \
    python3-pip \
    tar \
    gzip \
    make \
    wget \
    dnf-plugins-core

# Install GCC 10
RUN dnf -y install gcc-toolset-10-gcc gcc-toolset-10-gcc-c++

# Activate GCC 10 environment for subsequent commands
SHELL ["/usr/bin/scl", "enable", "gcc-toolset-10"]

# Verify the GCC version
RUN gcc --version && g++ --version

# Setup for CMake
COPY ./cmake-3.22.0-linux-x86_64.tar.gz /tmp/cmake.tar.gz
RUN mkdir /opt/cmake && \
    tar -xzvf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1
ENV PATH="/opt/cmake/bin:${PATH}"
RUN cmake --version

RUN dnf install -y binutils

# Clone your repository and build your application
WORKDIR /content
RUN git clone https://github.com/PrithivirajDamodaran/blitz-embed.git && \
    cd blitz-embed && \
    git submodule update --init --recursive
WORKDIR /content/blitz-embed
RUN cmake -B build . && make -C build -j

# Install utilities
WORKDIR /opt
RUN dnf install -y curl
RUN curl -L -o /opt/bge-base-en-v1.5-q4_0.gguf https://huggingface.co/prithivida/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_0.gguf

# Prepare the final image from almalinux:8 for compatibility
FROM almalinux:8
WORKDIR /app
COPY --from=build /content/blitz-embed/build/bin/* /app/
COPY --from=build /content/blitz-embed/build/src/libbert.so /app/
COPY --from=build /content/blitz-embed/build/ggml/src/libggml.so /app/
COPY --from=build /opt/bge-base-en-v1.5-q4_0.gguf /app/

# Adjust permissions
RUN chmod +x /app/*
RUN chmod 644 /app/libbert.so
RUN chmod 644 /app/libggml.so  
RUN chmod 644 /app/bge-base-en-v1.5-q4_0.gguf

ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

# Ensure your application listens on the PORT environment variable
CMD ["/app/encode"]
