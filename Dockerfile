FROM python:3.9.4-slim

RUN pip install --upgrade pip
RUN pip install requests
RUN pip install Flask
RUN pip install pyjwt

# Set the working directory to /app
WORKDIR /app

# Copy runtime files from the current directory into the container at /app
ADD simple_soe.py /app/
ADD tag-build.sh /app/

RUN mkdir /app/static
RUN /app/tag-build.sh > /app/static/build.txt
ADD static/favicon-96x96.png /app/static/

RUN ls -R
RUN cat /app/static/build.txt


EXPOSE 5040

# Run app.py when the container launches
CMD ["python3", "simple_soe.py"]
