# Faster Whisper FastAPI

🚀 Welcome to the Faster Whisper FastAPI project! This project is designed to provide a fast and efficient implementation of the Whisper algorithm using the FastAPI framework.

## Getting Started

🔍 To get started with the Faster Whisper FastAPI project, you can follow these steps:

1. Clone the project repository using the following command:

```
git clone https://github.com/frankwongWO/faster-whisper-fastapi.git
```

# colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/frankwongWO/7e1fb9de4ef6f14b8ebfc4f2f84251c8/faster-whisper-fastapi.ipynb)


# command

run

```
./run.ps1
```

deactivate

```
deactivate
```

# api docs

http://127.0.0.1:8123/docs#/

# install CUDA

https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

# install cuDNN

https://developer.nvidia.com/cudnn

# After a day of trial and error, I finally solved the issue!

For anyone having a problem, copy all DLLs from CUDNN, as well as cublasLt64_11.dll from the GPU Computing Toolkit into your ctranslate2 package directory. Since I'm using a venv, it was \faster-whisper\venv\Lib\site-packages\ctranslate2", but if you use Conda or just regular Python without virtual environments, it'll be different.

https://github.com/guillaumekln/faster-whisper/issues/85

# check cuda version

```
nvcc -V
```
