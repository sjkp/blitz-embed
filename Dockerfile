# Use the same build stage as before
FROM amazonlinux:2 as build

# Install development tools and additional dependencies
RUN yum install -y \
    git \
    libcurl-devel \
    openssl-devel \
    python3-pip \
    tar \
    gzip \
    make \
    wget \
    gcc-c++.x86_64 \
    binutils

# Install GCC 10
RUN yum -y install gcc10.x86_64 gcc10-c++.x86_64 && \
    ln -sf /usr/bin/gcc10 /usr/bin/gcc && \
    ln -sf /usr/bin/g++10 /usr/bin/g++

# Verify the GCC version
RUN g++ --version && gcc --version

# Setup for CMake
COPY ./cmake-3.22.0-linux-x86_64.tar.gz /tmp/cmake.tar.gz
RUN mkdir /opt/cmake && \
    tar -xzvf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1
ENV PATH="/opt/cmake/bin:${PATH}"
RUN cmake --version

# Clone your repository and build your application
WORKDIR /content
RUN git clone https://github.com/PrithivirajDamodaran/blitz-embed.git && \
    cd blitz-embed && \
    git submodule update --init --recursive
WORKDIR /content/blitz-embed
RUN cmake -B build . && make -C build -j

# Install utilities
WORKDIR /opt
RUN yum install -y curl
RUN curl -L -o /opt/bge-base-en-v1.5-q4_0.gguf https://huggingface.co/prithivida/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_0.gguf

# Prepare the final image
FROM amazonlinux:2
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

