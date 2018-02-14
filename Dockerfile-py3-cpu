FROM microsoft/cntk:2.4-cpu-python3.5

LABEL maintainer "MICROSOFT CORPORATION"

# Docker install
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common

RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# Object Detection
RUN apt-get update && apt-get install -y docker-ce && apt-get install -y --no-install-recommends \
        cmake \
        git \
        libopencv-dev \
        nvidia-cuda-toolkit \
        && \
    apt-get -y autoremove \
        && \
    rm -rf /var/lib/apt/lists/*

RUN /root/anaconda3/envs/cntk-py35/bin/conda install -y -n cntk-py35 cython boost
RUN bash -c 'source /cntk/activate-cntk && pip install dlib easydict pyyaml'
RUN bash -c 'source /cntk/activate-cntk && cd /cntk/Examples/Image/Detection/utils && git clone https://github.com/CatalystCode/py-faster-rcnn.git && cd py-faster-rcnn/lib && python setup.py build_ext --inplace'
RUN cp /cntk/Examples/Image/Detection/utils/py-faster-rcnn/lib/pycocotools/_mask.cpython-35m-x86_64-linux-gnu.so /cntk/Examples/Image/Detection/utils/cython_modules/ 
RUN cp /cntk/Examples/Image/Detection/utils/py-faster-rcnn/lib/utils/cython_bbox.cpython-35m-x86_64-linux-gnu.so /cntk/Examples/Image/Detection/utils/cython_modules/ 
RUN cp /cntk/Examples/Image/Detection/utils/py-faster-rcnn/lib/nms/gpu_nms.cpython-35m-x86_64-linux-gnu.so /cntk/Examples/Image/Detection/utils/cython_modules/ 
RUN cp /cntk/Examples/Image/Detection/utils/py-faster-rcnn/lib/nms/cpu_nms.cpython-35m-x86_64-linux-gnu.so /cntk/Examples/Image/Detection/utils/cython_modules/ 

WORKDIR /cntk/Examples/Image/Detection/FasterRCNN
RUN bash -c 'git config --system core.longpaths true'
RUN bash -c 'source /cntk/activate-cntk && pip install "git+https://github.com/Azure/azure-sdk-for-python#egg=azure-cognitiveservices-vision-customvision&subdirectory=azure-cognitiveservices-vision-customvision"'
COPY . /cv_workshop
WORKDIR /cv_workshop
RUN bash -c 'source /cntk/activate-cntk && pip install opencv-python'
RUN bash -c 'source /cntk/activate-cntk && pip install -r cli-requirements.txt'

RUN bash -c 'source /cntk/activate-cntk && python -u cvworkshop_utils.py'  

CMD bash -c 'source /cntk/activate-cntk && jupyter-notebook --ip=0.0.0.0 --allow-root --no-browser'

