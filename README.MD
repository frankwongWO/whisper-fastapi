# Faster Whisper FastAPI

🚀 Welcome to the Faster Whisper FastAPI project! This project is designed to provide a fast and efficient implementation of the Whisper algorithm using the FastAPI framework.

## Getting Started

🔍 To get started with the Faster Whisper FastAPI project, you can follow these steps:

1. Clone the project repository using the following command:

```
git clone https://github.com/frankwongWO/faster-whisper-fastapi.git
```
2. Install the required dependencies using the following command:
```
pip install -r requirements.txt
```
3. Start the FastAPI server using the following command:
```
uvicorn run:app --reload --port 8123
```
4. Open the API documentation in your web browser using the following URL:
http://localhost:8123/docs

# Requirements
* Python 3.6 or higher

# Installation
Before using Faster Whisper FastAPI, you need to install CUDA and cuDNN. Here are the installation instructions:

## Install CUDA
You can download the CUDA installer from the NVIDIA website. Here are the steps:

Go to the following link to download the CUDA installer:

https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

Run the CUDA installer and follow the instructions in the installation wizard.

## Install cuDNN
You can download the cuDNN files from the NVIDIA website. Here are the steps:

Go to the following link to download the cuDNN files:

https://developer.nvidia.com/cudnn

Extract the cuDNN files to a directory.

# Usage
To use the transcribe function, you need to send a POST request to the FastAPI server with the following parameters:

* model_size: a string specifying the size of the model to use. Valid values are "large-v2", "large-v1", "base", "tiny", "small", and "medium". You can find available models on the [Hugging Face Hub](https://huggingface.co/guillaumekln).
* device: a string specifying on which device to run the model. Valid values are "cpu" and "cuda". Defaults to "cpu".
* compute_type: a string specifying which compute type to use. Valid values are "int8", "int8_float16", and "float16". Defaults to "float16".
* to_lang: a string specifying the language to which the audio should be transcribed. Defaults to None.
* file: an uploaded file object containing the audio data to transcribe.

## Here is an example Python code to send a POST request:
```python
import requests

url = "http://localhost:8000/transcribe"

files = {"file": ("audio.mp3", open("audio.wav", "rb"))}
data = {
    "model_size": "base",
    "device": "cpu",
    "compute_type": "float16",
    "to_lang": "en",
}

response = requests.post(url, data=data, files=files)
print(response.json())

```
In the above example, we use the requests library to send a POST request to the FastAPI server. We specify the model size, device, and compute type to use, and upload the audio data to transcribe as a file. The server will return a JSON object containing the transcribed text result.

# Run in colab
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/frankwongWO/7e1fb9de4ef6f14b8ebfc4f2f84251c8/faster-whisper-fastapi.ipynb)


# Windows command

## run in windows
```shell
./run.ps1
```

## deactivate
```shell
deactivate
```

# Troubleshooting

For anyone having a problem, copy all DLLs from CUDNN, as well as cublasLt64_11.dll from the GPU Computing Toolkit into your ctranslate2 package directory. Since I'm using a venv, it was \faster-whisper\venv\Lib\site-packages\ctranslate2", but if you use Conda or just regular Python without virtual environments, it'll be different.

https://github.com/guillaumekln/faster-whisper/issues/85

## Check the CUDA version. Run the following command in the command line:
```
nvcc -V
```
# Reference
For more information on Faster Whisper FastAPI, please visit the following GitHub repository:

* [faster-whisper](https://github.com/guillaumekln/faster-whisper)
* [FastAPI documentation](https://fastapi.tiangolo.com/)
* [FastAPI GitHub repository](https://github.com/tiangolo/fastapi)


I hope this information is helpful to you!
