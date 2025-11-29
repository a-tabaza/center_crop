# Face Cropper

Stripped down functionality from [Face Crop Plus](https://github.com/mantasu/face-crop-plus) using [AMD's RetinaFace ONNX](https://huggingface.co/amd/retinaface) with no PyTorch dependancy, a 1.7 MB model artifact and a single file of code for inference.

# Why

What I need: square image of one face from image of one face that is not square.

What I don't need: 7GB venv with CUDA PyTorch dependencies.

# How

Detects bbox of a face (one face) in the picture (using RetinaFace and ONNX for inference) then crops a square around it with the face being in the center (cv2 slicing).

# Setup

Local

```bash
uv venv
uv pip install -r requirements.txt
. .venv/bin/activate
```
Docker (API)

Build

```bash
docker build -t cropper .
docker run -d -p 8080:8080 cropper:latest
```

Pull

```bash
docker pull ghcr.io/a-tabaza/center_crop:main
docker run -d -p 8080:8080 ghcr.io/a-tabaza/center_crop:main
```

Go to ``localhost:8080/docs`` for SwaggerUI

# Usage

CLI

```bash
python src/crop.py -i input.jpg -o output.jpg
```

API

Multiple images (result in .zip file)

```bash
curl -X 'POST' \
  'http://localhost:8080/crop/' \
  -H 'accept: image/jpeg' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@4.jpg;type=image/jpeg' \
  -F 'files=@3.jpeg;type=image/jpeg' \
  -F 'files=@2.jpg;type=image/jpeg' \
  -F 'files=@1.jpg;type=image/jpeg'
```

One image (result back as jpg)

```bash
curl -X 'POST' \
  'http://localhost:8080/crop/' \
  -H 'accept: image/jpeg' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@1.jpg;type=image/jpeg' \
```


# Examples
![](./demo_images/1.jpg)
![](./demo_images/1_crop.jpg)

![](./demo_images/2.jpg)
![](./demo_images/2_crop.jpg)

![](./demo_images/3.jpeg)
![](./demo_images/3_crop.jpg)

![](./demo_images/4.jpg)
![](./demo_images/4_crop.jpg)

# Contributing

No. This isn't for you, it's to save my time, if it helps, I'm glad, if it doesn't, fork and adapt it.