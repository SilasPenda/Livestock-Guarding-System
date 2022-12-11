# Livestock-Guard-System

## Inference test

Download weights

```
https://drive.google.com/file/d/1-FxeaU3uR_kKJ_6ufiEF9JJG1YMnekUU/view?usp=sharing
```

Create a Virtual environment.

```
pip install virtualenv
virtualenv -p /usr/bin/python3.8 [env-name]
```

Activate Virtual environment.

```
source [env-name]/bin/activate
```

Install requirements.

```
cd Livestock-Guarding-System
pip install -r requirements.txt
```

Inference using Native Pytorch using GPU

```
python3 detect.py --weights best.pt --conf 0.50 --source test_video.mp4 
```

Inference using Native Pytorch using CPU

```
python3 detect_without_jit.py --weights best.pt --conf 0.50 --source test_video.mp4 
```

Inference using Torch-ort using CPU \
NB: Linux machine only (Can use Colab) \

Access ORTModule

```
python3 -m onnxruntime.training.ortmodule.torch_cpp_extensions.install
```
Run Inference
```
python3 detect_ort.py --weights best.pt --conf 0.50 --source test_video.mp4 
```
