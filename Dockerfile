FROM ubuntu:18.04

# Dependencies for glvnd and X11.
RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Dependencies for python, opencv and vispy
RUN apt-get install -y -qq --no-install-recommends \
    python3 python3-pip libsm6 libxext6 libxrender-dev libglfw3 \
    libglib2.0-0 fontconfig \
    && python3 -m pip install setuptools vispy opencv-python
