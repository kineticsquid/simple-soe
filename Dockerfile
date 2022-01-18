FROM kineticsquid/simple-soe-base:latest

# Set the working directory to /app
WORKDIR /app

# Copy runtime files from the current directory into the container at /app
ADD *.py /app/

RUN mkdir /app/static
ADD static/ /app/static/
RUN mkdir /app/templates
ADD templates/ /app/templates/
RUN mkdir /app/tessdata
ADD tessdata/ /app/tessdata/
ADD tessdata/ /usr/share/tessdata/
ADD static/favicon-96x96.png /app/static/
RUN date > /app/static/build.txt

# Run app.py when the container launches
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 simple-soe:app
