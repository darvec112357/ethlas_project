# start by pulling the python image
FROM python:3.8-slim

# switch working directory
WORKDIR /app

# copy every content from the local file to the image
COPY . /app

RUN python -m pip cache purge

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run app.py when the container launches
CMD ["python", "app.py"]
