FROM pytorchlightning/pytorch_lightning
RUN apt-get update & apt-get upgrade
COPY . /app
WORKDIR /app/src
RUN pip3 install seaborn
ENTRYPOINT ["python3", "main.py"]
