FROM pytorchlightning/pytorch_lightning
RUN apt-get update & apt-get upgrade
COPY . /app
WORKDIR /app/src
CMD python3 train.py
