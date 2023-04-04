# run command

./run.ps1

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

nvcc -V
