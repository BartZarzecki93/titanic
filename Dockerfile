# Set container image
FROM python:3.6-stretch

# Check our python environment
RUN python3 --version
RUN pip3 --version

# Working directory
WORKDIR /

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY src/ /src/
COPY initial_data/ /data/
COPY main.py /

# Check if all files were copied
RUN ls -la /src/*
RUN ls -la
RUN cd data && ls -la

# Running the app
CMD ["python3", "main.py"]