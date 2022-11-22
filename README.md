# Livestock-Guard-System

## Test 

NB: Linux machine only (Can use Colab)

Download weights

```
https://drive.google.com/file/d/124TxRpALLUb7zvGgOPpRs6JD5dcrP4_z/view?usp=sharing
```

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
python3 detect_without_jit.py --weights best_292.pt --conf 0.50 --source test_video.mp4 
```

Inference using Torch-ort

```
python3 detect_ort.py --weights best_292.pt --conf 0.50 --source test_video.mp4 
```
