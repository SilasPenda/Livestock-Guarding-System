# Livestock-Guard-System

## Test 

Create a Virtual environment.

```
pip install virtualenv
virtualenv -p /usr/bin/python2.8 [env-name]
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

Inference using Native Pytorch

```
python3 detect_without_jit.py --weights runs/train/best.pt --conf 0.25 --source test_video.mp4 
```

Inference using Torch-ort

```
python3 detect_ort.py --weights runs/train/best.pt --conf 0.25 --source test_video.mp4 
```
