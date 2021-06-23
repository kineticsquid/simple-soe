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


ADD tag-build.sh /app/

RUN date > /app/static/build.txt
ADD static/favicon-96x96.png /app/static/

RUN ls -R
RUN cat /app/static/build.txt


EXPOSE 5040

# Run app.py when the container launches
CMD ["python3", "simple_soe.py"]
