
import core
from fastapi import FastAPI

app = FastAPI()


# Welcome
@app.get('/')
async def root():
    return {'prediction': 'Welcome to the EvoDrone video segmentation API'}


# Individual frame prediction
@app.post('/predict_frame')
async def predict_frame():
    return core.predict_api()
    # return {'prediction': 'pred'}


# Batch prediction
@app.post('/predict_batch')
async def predict_batch():
    return {'prediction': 'pred'}


# Full video prediction / Data upload
@app.post('/predict_batch')
async def predict_full_video():
    return {'prediction': 'pred'}


# Model information
# returns the model’s name, parameters and both the categorical and numerical features used in training.
@app.post('/model_info')
async def model_info():
    return {'model info': '...'}


# Health checks