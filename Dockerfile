FROM ubuntu:22.04




LABEL maintained="datolg123@gmail.com"
LABEL version="1.0"
LABEL decription='This is an example of dockerfile for Federated  Learning'
LABEL name="fed_2024"


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install curl -y
#confirm that have the prerequisites installed
RUN apt-get install -y software-properties-common 
# need PPA. add maintained by deadsnakes
RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt update
# install python 
RUN apt-get install -y python3.9  
# install package pip for ubunt
RUN apt-get install -y python3-pip 
RUN pip3 install --upgrade pip
RUN apt-get update
RUN apt -y upgrade 


WORKDIR /fed_2024
COPY . /fed_2024/

RUN pip install -r /fed_2024/requirements.txt

RUN python3 /fed_2024/dataset/generate_Cifar10.py noniid  
RUN bash /fed_2024/run_kdx.sh