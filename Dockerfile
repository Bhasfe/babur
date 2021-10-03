FROM continuumio/miniconda:latest

WORKDIR /app

COPY environment.yml ./
COPY .env ./.env
COPY bot.py ./
COPY run.sh ./
COPY ./data ./data
COPY ./model ./model
COPY ./logs ./logs
COPY ./channels_info.json ./channels_info.json

RUN ["chmod", "+x", "./run.sh"]

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "babur", "/bin/bash", "-c"]

ENV PATH /opt/conda/envs/babur/bin:$PATH

EXPOSE 5000

RUN source activate babur

ENTRYPOINT ["python"]
CMD ["bot.py"]
