<p align="center">
  <img alt="Blitz-Embed Library Icon" src="logo2.png" width="100%">
</p>

### Status - \[updated 3rd March 2024\]


| Serverless Provider | Dev Status | Provider billing logic |        Details          |
|----------|:---------------------:|:--------------------------|--------------------------|
Google Cloud Run C++ Wrappers | ✅ | Runtime **AND** Allocated Memory| You can choose CPU and Memory seperately |
AWS Lambda C++ Wrappers | ✅ |Runtime **X** Allocated Memory| You can choose only Memory |
Azure Functions C++ Wrappers | WIP | 
Google Cloud functions C++ Wrappers | ⛔ | 


### **Some numbers on what can you run for free:** 
----


**AWS** (has a free quota of 1M req /mo and 400,000 GB-sec).

| Batch Size | Tokens per text | Time (ms) | CPU/Mem | # Embeddings in Free quota | Cost After Quota         |
|------------|------------------|-----------|------|:-----------------------:|--------------------------|
| 1          | 512              | ~1100     | 6/10GB | 37K texts (18M free tokens)                   | $0.36 / Million tokens   |
| 6          | 64               | ~750      | 6/10GB | 60K   (30M texts free tokens)                 | $0.291 / Million tokens  |

- *tests ran on 4-bit GGML bge-base-en-v1.5*
- *pricing based on ap-south1(mumbai)*

**Google cloud run** (has a free quota of 2 M req per mon, 360K GB-seconds of memory per mon and 180K vCPU-sec).

| Batch Size | Tokens per text | Time (ms) | CPU/Mem | # Embeddings in Free quota | Cost After Quota         |
|------------|------------------|-----------|------|:-----------------------:|--------------------------|
| 1          | 512              | ~1300     | 8/4GB | 17K texts (8M free tokens)                 | $0.51 / Million tokens   |
| 6          | 64               | ~900      | 8/4GB | 25K  texts (12.8M free tokens)                 | $0.35 / Million tokens  |

- *tests ran on 4-bit GGML bge-base-en-v1.5*
- *pricing based on us-central1*

**References:**

- [AWS lambda pricing](https://lnkd.in/eu6k4e-3)
- [Azure functions pricing](https://azure.microsoft.com/en-us/pricing/details/functions/)


## What is it ?
C++ inference wrappers for running blazing fast embedding services on your favourite serverless.

- Leverages`GGML BERT` implementations.
- Bare-metal performance with e2e C++, No Python bindings.
- Speed without compromise of quality.(See benchmarks)
- Scale (`DIY Socket servers`, `vanilla HTTP` or `gRPC` deployments are no match to the scale or $ of Serverless like AWS lambda. ).
- Quantisation options - `q8_0`, `q5_0`, `q5_1`, `q4_0`, and `q4_1`.
- Super Economical as you pay / invocations that are tiny + quicker runtimes.
- Supports `BERT based embedders`. (Any lang)
- Smart folks have tested on: `BAAI/bge* models like bge-base-en-v1.5 and Sentence Transformers/all-MiniLM* models like all-MiniLM-L6-v2`.
- [Pre-quantised models to get started](https://huggingface.co/collections/prithivida/gguf-models-65e12c930890daf03e7e42ea).
- Optionally deploy on CUDA infra for GPU support.
- Forked with thanks from [bert GGML + python bindings](https://github.com/iamlemec/bert.cpp), [bert.cpp](https://github.com/skeskinen/bert.cpp), [embeddings.cpp](https://github.com/xyzhang626/embeddings.cpp)

## Who is it for ?
- Any one who wants to run a perfomant / cost efficent embedding service on SoTA embedders.
- Fair warning: Learning curve can be a little steep for absolute beginners.
 

### Contributions:
- C++ AWS Lambda handler for GGML bert. 
- C++ Google Cloud Run handler for GGML bert. 
- Prepackaged Dockerfiles for AWS Lambda.
- Prepackaged Dockerfiles for Google cloud run.
- GGUF files in `HuggingFace`.

### Roadmap
<details open>
<summary>Features</summary>

- C++ Azure functions handler + Docker file.
- Add support for embeddders like BGE-M3, allmpnet, SPLADE models.
- Add support for Matryoshka embeddings.
- Extend GPU support for standalone deployments.
- Bring in developments embedding related from llama.cpp.
</details>

### Why 4-bit quantisation is recommended ?
`Quantisation Jesus Tim Dettmers` has argued in the [15th min of this video](https://www.youtube.com/watch?v=y9PHWGOa8HA) and in this [paper](https://arxiv.org/pdf/2212.09720.pdf)
that 4-bit quantisation yields "best bit by bit performance" for a model.

### How Install & launch a embedding service as Google Cloud Run service?
<details>
<summary>Steps Involved</summary>

```sh

# 1.Clone repo

git clone https://github.com/PrithivirajDamodaran/blitz-embed.git
cd blitz-embed
mv Dockerfile-gcr Dockerfile

# 2. Setup Serverless for Google Cloud if you haven't

    Get your google project id

# 3. Run
```

```sh
# ensure docker daemon is running
docker build --no-cache --platform linux/amd64 -t gcr.io/<your_project_id>/blitz-embed:v1 .
gcloud auth configure-docker
gcloud auth login
docker push gcr.io/<your_project_id>/blitz-embed:v1
# verify image in cloud console
gcloud run deploy blitz-embed --image gcr.io/<your_project_id>/blitz-embed:v1 --platform managed --region <your-region> --allow-unauthenticated --memory=4Gi --cpu=8 --project <your_project_id> --concurrency=10 
#--allow-unauthenticated is only for testing, you need to protect your end point

# You will get an endpoint like https://blitz-embed-<get_your_own>.run.app

```
</details>

### Calling AWS Lambda embedding service in your app
<details>
<summary>Python snippet</summary>

```python
import requests
import json
import time
import numpy as np

url = 'https://blitz-embed-<get_your_own>.run.app'
payload = {
    "sent": [
            "All technical managers must have hands-on experience. For example, managers of software teams must spend at least 20% of their time coding. Solar roof managers must spend time on the roofs doing installations. Otherwise, they are like a cavalry leader who can't ride a horse or a general who can't use a sword.",
            "It's OK to be wrong. Just don't be confident and wrong.",
            "Never ask your troops to do something you're not willing to do.",
            "The only rules are the ones dictated by the laws of physics. Everything else is a recommendation.",
            "When hiring, look for people with the right attitude. Skills can be taught. Attitude requires a brain transplant.",
            "Whenever there are problems to solve, don't just meet with your managers. Do a skip level, where you meet with the right below your managers."
        ],
    "model": "/opt/bge-base-en-v1.5-q4_0.gguf", 
    "batch_size": 6,
    "max_len": 64,
    "normalise": True,
}

resp = requests.post(url=url, json=payload)
resp_obj = resp.json()
resp_body = json.loads(resp_obj["body"])

embeds = json.loads(resp_body["embedding"])
emb = np.array(embeds, dtype="float32")
print("Tokenisation and Inference time", round(resp_body["itime"], 1) * 0.001, " ms") # / 1000 as I am returning time in microseconds
```
</details>




### How Install & launch a embedding service as AWS Lambda?
<details>
<summary>Steps Involved</summary>

```sh

# 1.Clone repo

git clone https://github.com/PrithivirajDamodaran/blitz-embed.git
cd blitz-embed
mv Dockerfile-aws Dockerfile

# Goto src/CMakeLists.txt under "# main entry"
Uncomment # add_executable(encode run_aws.cpp)   
Comment   add_executable(encode run_gcr.cpp)
Uncomment # target_link_libraries(encode PRIVATE bert ggml aws-lambda-runtime curl)
Comment target_link_libraries(encode PRIVATE bert ggml curl)

# 2. Setup Serverless for AWS if you haven't
```

    Goto AWS IAM user Dashboard
    Create or reuse a user.
    Add AdministratorAccess permissions tab
    Get your_key and your_secret from Security Credentials tab


```sh
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

Use npm install -g serverless 


# 3. Run
# ensure docker daemon is running
serverless deploy
```
</details>

### Calling AWS Lambda embedding service in your app
<details>
<summary>Python snippet</summary>

```python
import requests
import json
import time
import numpy as np

url = 'https://your-service-url.amazonaws.com/encode'
payload = {
    "sent": [
        "It's OK to be wrong. Just don't be confident and wrong.",
        "Never ask your troops to do something you're not willing to do.",
        "The only rules are the ones dictated by the laws of physics. Everything else is a recommendation.",
        "When hiring, look for people with the right attitude. Skills can be taught. Attitude requires a brain transplant.",
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
</details>

### Exporting and Quantising Huggingface models

#### Install 
```python
git clone https://github.com/PrithivirajDamodaran/blitz-embed.git
cd blitz-embed
git submodule update --init --recursive
pip install -r requirements.txt
cmake -B build . && make -C build -j
```
#### Convert

default is f32, for f16 you need to as pass param as below

```python
mkdir models
python blitz-embed/convert.py BAAI/bge-base-en-v1.5 models/bge-base-en-v1.5-f16.gguf f16
```

#### Quantize
You need to pass any one of the options - `q8_0`, `q5_0`, `q5_1`, `q4_0`, and `q4_1`.

```python
build/bin/quantize models/bge-base-en-v1.5-f32.gguf models/bge-base-en-v1.5-q4_0.gguf q4_0
```

### MTEB benchmarks
<details>
<summary>Numbers</summary>


Legacy MTEB scores, Consolidated from other forks for reference.

MTEB (Massive Text Embedding Benchmark) for GGUF bert.cpp models vs. [sbert](https://sbert.net/) on CPU. All these benchmarks were run batchless before, the latest fork i.e this one supports batch inference.

### all-MiniLM-L6-v2
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| GGUF f32 | 0.8201 | 6.83 | 0.4082 | 11.34 | 
| GGUF f16 | 0.8201 | 6.17 | 0.4085 | 10.28 | 
| GGUF q4_0 | 0.8175 | 5.45 | 0.3911 | 10.63 | 
| GGUF q4_1 | 0.8223 | 6.79 | 0.4027 | 11.41 | 
| Vanilla sbert-batchless | 0.8203 | 13.10 | 0.4085 | 15.52 | 

### all-MiniLM-L12-v2
| Data Type | STSBenchmark | eval time | EmotionClassification | eval time | 
|-----------|-----------|------------|-----------|------------|
| GGUF f32 | 0.8306 | 13.36 | 0.4117 | 21.23 | 
| GGUF f16 | 0.8306 | 11.51 | 0.4119 | 20.08 | 
| GGUF q4_0 | 0.8310 | 11.27 | 0.4183 | 20.81 | 
| GGUF q4_1 | 0.8325 | 12.37 | 0.4093 | 19.38 | 
| Vanilla sbert-batchless | 0.8309 | 22.81 | 0.4117 | 28.04 | 

### BGE_base_en_v1.5
| Data Type | STSBenchmark | eval time | 
|-----------|-----------|------------|
| GGUF f32 | 0.8530 | 20.04 | 
| GGUF f16 | 0.8530 | 21.82 | 
| GGUF q4_0 | 0.8509 | 18.78 | 
| GGUF q4_0-batchless | 0.8509 | 35.97 |
| GGUF q4_1 | 0.8568 | 18.77 |
| Vanilla sbert-batchless | 0.8464 | 64.58 | 

</details>