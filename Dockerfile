FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN pip install --upgrade pip

RUN pip install \
    Pillow==6.2.1 \
    scipy==1.2.0 \
    wandb

RUN pip install hydra-core==1.0.6
RUN pip install fvcore==0.1.5.post20210423
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/cu110/torch_stable.html
RUN pip install classy_vision==0.6.0
RUN pip install fairscale==0.1.0
