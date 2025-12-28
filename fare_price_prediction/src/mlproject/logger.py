
import os
import logging
from datetime import datetime


# creates the log file name, and, logs folder where we can store all log files
LOG_FILE = f"{datetime.now().strftime("%d_%m_%Y_%H_%M_%S")}.log"
logs_path = os.path.join( os.getcwd(), "logs" )

os.makedirs(logs_path, exist_ok=True)


# creating the log file path for configuration
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)


# configuring how logging works -> upon using logging.info() this style entry pass in the mentioned log file
logging.basicConfig(
    # where the output should go
    filename = LOG_FILE_PATH,

    # how the output should look
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",

    # which logs are recorded
    level = logging.INFO

)

if __name__=="__main__":
    print("project executed from logging.info")
    logging.info("logging.info runs successfully!")
    print(LOG_FILE)
    print(logs_path)
    print(LOG_FILE_PATH)