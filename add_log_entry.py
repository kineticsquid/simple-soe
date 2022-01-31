import json

def log(log_message, flask_app=None):
    if flask_app is not None:
        flask_app.logger.info(log_message)
    else:
        print(log_message)