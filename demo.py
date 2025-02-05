import os

from us_visa.logger import logging
from us_visa.exception import USvisaException
import sys



'''
logging.info("Welcome to our custom logs")


try:
    a = 2/0
    
except Exception as e:
    raise USvisaException(e,sys)

'''
'''
from us_visa.constants import MONGODB_URL_KEY
mongo_db_url = os.getenv(MONGODB_URL_KEY)
print(mongo_db_url)

'''
from us_visa.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()
