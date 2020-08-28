FROM python:3.6-stretch

# check our python environment
RUN python3 --version
RUN pip3 --version

WORKDIR /src

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
RUN ls -la /src/*

# Running Python Application
CMD ["python3", "/src/main.py"]