FROM nvcr.io/nvidia/isaac-lab:3.0.0
ARG PY=/opt/isaaclab-venv/bin/python
RUN ($PY -m pip --version || $PY -m ensurepip) \
 && $PY -m pip freeze | grep -iE '^(numpy|torch|torchvision|warp-lang|pillow)==' > /tmp/keep.txt \
 && $PY -m pip install --no-cache-dir -c /tmp/keep.txt "timm>=1.0.27,<1.1" "huggingface-hub>=1.0" "av>=14.1" click \
      pillow-heif matplotlib embreex==4.4.0.post1 "onnxruntime>=1.20,<2" pyserial opencv-python-headless scipy pyyaml \
 && $PY -m pip install --no-cache-dir --no-deps \
      "depth_pro_2 @ git+https://github.com/apple-aiml-research/ml-depth-pro-2@aa389cbb0b74341664ed663a0966c859905cd6c0" \
 && $PY -c "import depth_pro_2, embreex, onnxruntime, timm, cv2, serial; print('CMDTEST_IMAGE_OK')"
