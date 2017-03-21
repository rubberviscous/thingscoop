FROM ubuntu:16.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy \
        ffmpeg \
        imagemagick \
        ghostscript && \
    rm -rf /var/lib/apt/lists/*
RUN pip install pyyaml
RUN pip install docopt
RUN pip install opencv-python
RUN pip install progressbar2
RUN pip install requests
RUN pip install moviepy
RUN pip install pattern
RUN pip install termcolor
RUN pip install pypeg2
# RUN pip install git+git://github.com/rubberviscous/thingscoop.git@master_dd
ADD . /home/docker/code
ENV PATH="/home/docker/code/bin:${PATH}"
RUN mkdir -p /app/googlenet_imagenet
WORKDIR /app