
<p align="center">
  <img alt="Blitz-Embed Library Icon" src="logo.png" width="800" height="100">
</p>

## What is it ?
C++ inference wrappers for running blazing fast embedding services on your favourite serverless.

- Leverages`GGML BERT` implementations - forked with thanks from [bert.cpp + python bindings](https://github.com/iamlemec/bert.cpp), [bert.cpp](https://github.com/skeskinen/bert.cpp), [embeddings.cpp](https://github.com/xyzhang626/embeddings.cpp)
- Bare-metal performance with e2e C++, No Python bindings.
- Speed without compromise of quality.(See benchmarks)
- Scale (`DIY Socket servers`, `vanilla HTTP` or `gRPC` deployments are no match to scale of Serverless like AWS lambda. Serverless also comes with great bells & whistles).
- Quantisation options - `q8_0`, `q5_0`, `q5_1`, `q4_0`, and `q4_1`.
- Super Economical $ compared to ONNX (owing to smaller size, quicker runtime)
- Supports any BERT based embedders. 
- Tested on: 

## Who is it for ?
- Any one who wants to run a perfomant / cost efficent embedding service on SoTA embedders.
- Fair warning: Learning curve can be a little steep for absolute beginners.
 

### Contributions:
- C++ AWS Lambda handler for GGML bert. [[Jump to "Get started"]]()
- Prepackaged Dockerfiles for AWS Lambda.
- GGUF files in `HuggingFace`.


### How to launch a embedding service in AWS Lambda

    Steps Involved

```sh

# 1.Clone repo

git clone https://github.com/PrithivirajDamodaran/blitz-embed.git
cd blitz-embed

# 2. Setup Serverless for AWS if you haven't
```

    `AWS IAM user Dashboard`, 
    Create or reuse a user.
    Add `AdministratorAccess` permissions tab
    Get your_key and your_secret from `Security Credentials` tab


```sh
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

Use npm install -g serverless 


# 3. Run
# ensure docker daemon is running
serverless deploy
```


### Calling embedding service in your app

```python
import requests
import json
import time
import numpy as np

url = 'https://your-service-url.amazonaws.com/encode'
payload = {
    "sent": [
        "A",
        "B",
        "C"
    ],
    "model": "/opt/bge-base-en-v1.5-q4_0.gguf",
    "batch_size": 4,
    "max_len": 256,
    "normalise": True,
}

resp = requests.post(url=url, json=payload)
resp_obj = resp.json()
embeds = json.loads(resp_obj["embedding"])
emb = np.array(embeds, dtype="float32")
print("Tokenisation and Inference time", round(resp_obj["itime"], 1) * 0.001, " ms") # / 1000 as this time comes in microseconds
```


### Roadmap
- C++ GCP functions handler + Docker file.
- C++ Azure functions handler + Docker file.
- Add support for embeddders like BGE-M3, allmpnet, SPLADE models.


### MTEB benchmarks

From other forks



### Install

Fetch this repository then download submodules and install packages with
```sh
git submodule update --init
pip install -r requirements.txt
```

To fetch models from `huggingface` and convert them to `gguf` format run something like the following (after creating the `models` directory)
```sh
python bert_cpp/convert.py BAAI/bge-base-en-v1.5 models/bge-base-en-v1.5-f16.gguf
```
This will convert to `float16` by default. To do `float32` add `f32` to the end of the command.

### Build

To build the C++ library for CPU/CUDA/Metal, run the following
```sh
# CPU
cmake -B build . && make -C build -j

# CUDA
cmake -DGGML_CUBLAS=ON -B build . && make -C build -j

# Metal
cmake -DGGML_METAL=ON -B build . && make -C build -j
```
On some distros, when compiling with CUDA, you also need to specify the host C++ compiler. To do this, I suggest setting the `CUDAHOSTCXX` environment variable to your C++ bindir.

