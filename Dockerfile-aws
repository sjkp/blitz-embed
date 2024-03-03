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

# Install GCC 10
RUN yum -y install gcc10.x86_64 gcc10-c++.x86_64

RUN ln -sf /usr/bin/gcc10-gcc /usr/bin/gcc && \
    ln -sf /usr/bin/gcc10-g++ /usr/bin/g++

# Verify the GCC version
RUN g++ --version && gcc --version

# Copy the local CMake tar.gz file into the container
COPY ./cmake-3.22.0-linux-x86_64.tar.gz /tmp/cmake.tar.gz

# Create the directory, extract the file to /opt/cmake, and adjust the directory structure as needed
RUN mkdir /opt/cmake && \
    tar -xzvf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1


# Add CMake to the PATH
ENV PATH="/opt/cmake/bin:${PATH}"

# Verify the installation
RUN cmake --version


RUN yum install -y binutils

# Clone and build AWS Lambda C++ runtime library
WORKDIR /build
RUN git clone https://github.com/awslabs/aws-lambda-cpp-runtime.git && \
    cd aws-lambda-cpp-runtime && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make && make install

# Copy the static library to /usr/local/lib
RUN cp /build/aws-lambda-cpp-runtime/build/libaws-lambda-runtime.a /usr/local/lib/

# Copy header files to /usr/local/include, creating a subdirectory for them
RUN mkdir -p /usr/local/include/aws-lambda-runtime && \
    cp -r /build/aws-lambda-cpp-runtime/include/* /usr/local/include/aws-lambda-runtime/

# Optionally, copy any additional headers that might be needed, adjusting paths as necessary
# RUN cp /build/aws-lambda-cpp-runtime/src/backward.h /usr/local/include/aws-lambda-runtime/


RUN ls -l /usr/local/lib
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

# Clone your repository
WORKDIR /content
RUN git clone https://github.com/PrithivirajDamodaran/blitz-embed.git && \
    cd blitz-embed && \
    git submodule update --init --recursive

# Build your AWS Lambda C++ application that includes `BertApp`
WORKDIR /content/blitz-embed
RUN cmake -B build . && make -C build -j

# Install curl or any other utilities you need
WORKDIR /opt
RUN yum install -y curl
RUN curl -L -o /opt/bge-base-en-v1.5-q4_0.gguf https://huggingface.co/prithivida/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-q4_0.gguf
# RUN curl -L -o /opt/bge-base-en-v1.5-f32.gguf https://huggingface.co/prithivida/bge-base-en-v1.5-gguf/resolve/main/bge-base-en-v1.5-f32.gguf


# Copy the built application to the lambda runtime base image
FROM amazonlinux:2
COPY --from=build /content/blitz-embed/build/bin/encode /var/task/encode
COPY --from=build /content/blitz-embed/build/src/libbert.so /var/task/libbert.so
COPY --from=build /content/blitz-embed/build/ggml/src/libggml.so /var/task/libggml.so
COPY --from=build /opt/bge-base-en-v1.5-q4_0.gguf /opt/bge-base-en-v1.5-q4_0.gguf
# COPY --from=build /opt/bge-base-en-v1.5-f32.gguf /opt/bge-base-en-v1.5-f32.gguf

RUN chmod +x /var/task/encode
RUN chmod 644 /var/task/libbert.so
RUN chmod 644 /var/task/libggml.so  

RUN chmod 644 /opt/bge-base-en-v1.5-q4_0.gguf
# RUN chmod 644 /opt/bge-base-en-v1.5-f32.gguf


ENV LD_LIBRARY_PATH=/var/task:$LD_LIBRARY_PATH

# Set the handler function name
CMD ["/var/task/encode"]


