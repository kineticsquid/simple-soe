# Using 4.1.0 explicitly because as of 6/15, 4.1.1 did not have required certs installed
FROM kineticsquid/sudoku-bot-base:latest

# Set the working directory to /app
WORKDIR /app

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

RUN date > /app/static/build.txt

# Run app.py when the container launches
ENTRYPOINT ["python", "app.py"]
