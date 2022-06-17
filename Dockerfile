FROM python:3.9.12
COPY requirements.txt /tmp
COPY . /app
WORKDIR /app/site
EXPOSE 8501
RUN python -m pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]