# FROM continuumio/miniconda3
FROM python:3.10

WORKDIR /SystemCode

COPY ./app/requirements.txt .
# RUN conda env create -f environment.yml
RUN pip install -r requirements.txt

COPY . .

# Make RUN commands using the new environment:
# RUN echo "conda activate faq_bot" >> ~/.bashrc
# SHELL ["/bin/bash", "--login", "-c"]

EXPOSE 7868
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]


## Docker Instructions ##
# docker build . -t parliagpt:latest
# docker run -dp 7868:7868 --gpus all --name parliagpt_app parliagpt
# Access via http://localhost:7860