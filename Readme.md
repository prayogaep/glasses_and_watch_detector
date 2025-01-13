# Glasses and Watch Object Detector

How to use ?

## Install requirements.txt

Execute script on terminal
```
pip install -r requirements.txt
```


## Create Virtual Environtment

Execute script on terminal

```
python -m venv venv
```

## Activate Environment (Windows)

Execute script on terminal

```
.\venv\Scripts\activate
```

### Optional
Execute script on powershell if activate venv is failed. After Execute this script you can try again execute activate script.
```
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### If Success

Will display ```(venv) C:\path\to\object_detection_app>```

on your terminal



# Run Video Processing

Execute script on terminal
```
python .\src\main_open_cv2.py
```

# Run Image Processing

Execute script on terminal
```
python .\src\main_image_processing.py
```