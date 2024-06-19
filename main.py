from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator, ValidationError
import tensorflow as tf
import numpy as np
import pandas as pd
from google.cloud import firestore
from google.cloud import storage
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import logging
import sys
import os
from typing import List
from cachetools import TTLCache
from datetime import datetime, timedelta

# In-memory cache with TTL (time-to-live)
cache_ttl_seconds = 1800  # Cache TTL in seconds (e.g., 5 minutes)
cities_cache = TTLCache(maxsize=100, ttl=cache_ttl_seconds)
categories_cache = TTLCache(maxsize=100, ttl=cache_ttl_seconds)


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(asctime)s [%(processName)s: %(process)d] [%(threadName)s: %(thread)d] [%(levelname)s] %(name)s: %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# Firestore client initialization
logger.info("Initializing Firestore client")
try:
    db = firestore.Client()
    logger.info("Firestore client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {e}")
    sys.exit(1)

# Initialize TF-IDF Vectorizer and OneHotEncoder
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
onehot_encoder = OneHotEncoder(sparse_output=False)

# Function to download model from GCS
def download_model_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f'Model downloaded from GCS bucket {bucket_name}, blob {source_blob_name} to {destination_file_name}')

# Download and load the model at startup
bucket_name = 'eventure-bucket-2024'
model_blob_name = 'models/models_lat_long_city.h5'
local_model_path = 'models/models_lat_long_city.h5'

if not os.path.exists(local_model_path):
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
    download_model_from_gcs(bucket_name, model_blob_name, local_model_path)
else:
    logger.info(f'Model already exists at {local_model_path}, skipping download.')

model = tf.keras.models.load_model(local_model_path)

app = FastAPI()

logger.info('API is starting up')

class RecommendRequest(BaseModel):
    Category: List[str]
    Location_City: str
    
    @validator('Category')
    def check_category_list(cls, v):
        if not isinstance(v, list):
            raise ValueError('Category must be a list')
        return v

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Validation error",
            "data": exc.errors()
        },
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "message": "Validation error",
            "data": exc.errors()
        },
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": str(exc),
            "data": []
        },
    )

@app.post("/recommend")
async def recommend_events(request: RecommendRequest):
    logger.info("Received request for /recommend endpoint")
    try:
        input_categories = request.Category
        input_location = request.Location_City
        
        if not isinstance(input_categories, list):
            raise HTTPException(status_code=400, detail="Category must be a list")
        
        # Log input dimensions
        logger.info(f'Input categories: {input_categories}')
        logger.info(f'Input location: {input_location}')

        # Fetch data from Firestore
        logger.info("Fetching events from Firestore")
        events_ref = db.collection('events')
        events_docs = events_ref.stream()
        events_data = []
        for doc in events_docs:
            events_data.append(doc.to_dict())
        
        if not events_data:
            raise HTTPException(status_code=404, detail="No events found in the database")
        
        # Convert events data to DataFrame
        data = pd.DataFrame(events_data)
        
        # Preprocess the data
        data['Category'] = data['Category'].fillna('')
        data['Location_City'] = data['Location_City'].fillna('')
        
        # Fit TF-IDF and OneHotEncoder on the entire data
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['Category'])
        location_matrix = onehot_encoder.fit_transform(data[['Location_City']])
        combined_matrix = np.hstack((tfidf_matrix.toarray(), location_matrix))

        logger.info(f'combined_matrix shape: {combined_matrix.shape}')

        # Process input
        category_vecs = tfidf_vectorizer.transform(input_categories)
        category_vec = np.mean(category_vecs, axis=0).reshape(1, -1)
        location_vec = onehot_encoder.transform([[input_location]])

        logger.info(f'category_vec shape: {category_vec.shape}')
        logger.info(f'location_vec shape: {location_vec.shape}')

        combined_input = np.hstack((category_vec, location_vec))

        logger.info(f'combined_input shape: {combined_input.shape}')
        
        # Ensure encoded_input has the correct shape
        encoded_input = model.predict(combined_input)
        encoded_input = np.squeeze(encoded_input)  # Remove unnecessary dimensions
        encoded_input = encoded_input.reshape(1, -1)  # Reshape to (1, 94)

        logger.info(f'encoded_input shape: {encoded_input.shape}')

        # Ensure both arrays have the same number of dimensions
        if len(encoded_input.shape) != len(combined_matrix.shape):
            logger.error(f'Mismatched dimensions: encoded_input shape: {encoded_input.shape}, combined_matrix shape: {combined_matrix.shape}')
            raise HTTPException(status_code=500, detail="Input arrays have mismatched dimensions")

        cosine_similarities = linear_kernel(encoded_input, combined_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[::-1]

        logger.info(f'cosine_similarities shape: {cosine_similarities.shape}')
        logger.info(f'related_docs_indices: {related_docs_indices}')

        recommended_events = []
        seen_titles = set()
        max_events = 20

        for idx in related_docs_indices:
            event = data.iloc[idx]
            if event['Category'] in input_categories and event['Location_City'] == input_location:
                if len(recommended_events) < max_events:
                    recommended_events.append({
                        'Title': event['Title'],
                        'Category': event['Category'],
                        'Event_Start': event['Event_Start'],
                        'Location_City': event['Location_City'],
                        'Location_Address': event['Location_Address']
                    })
                    seen_titles.add(event['Title'])

        for idx in related_docs_indices:
            event = data.iloc[idx]
            if event['Category'] in input_categories and event['Location_City'] != input_location:
                if len(recommended_events) < max_events:
                    recommended_events.append({
                        'Title': event['Title'],
                        'Category': event['Category'],
                        'Event_Start': event['Event_Start'],
                        'Location_City': event['Location_City'],
                        'Location_Address': event['Location_Address']
                    })
                    seen_titles.add(event['Title'])

        additional_events = []
        for idx in related_docs_indices:
            event = data.iloc[idx]
            if event['Category'] not in input_categories and event['Title'] not in seen_titles:
                additional_events.append({
                    'Title': event['Title'],
                    'Category': event['Category'],
                    'Event_Start': event['Event_Start'],
                    'Location_City': event['Location_City'],
                    'Location_Address': event['Location_Address']
                })
                seen_titles.add(event['Title'])
            if len(additional_events) >= 5:
                break

        recommended_events.extend(additional_events[:5])

        return {
            "success": True,
            "message": "Get Recommendation by Categories and Location",
            "data": recommended_events
        }
    except HTTPException as http_exc:
        logger.error(f'HTTPException in recommend_events: {http_exc.detail}')
        return {
            "success": False,
            "message": http_exc.detail,
            "data": []
        }
    except Exception as e:
        logger.error(f'Error in recommend_events: {e}')
        return {
            "success": False,
            "message": str(e),
            "data": []
        }

@app.get("/cities")
async def get_cities():
    logger.info("Received request for /cities endpoint")
    try:
        if 'cities' in cities_cache:
            logger.info("Returning cached cities data")
            return {
                "success": True,
                "message": "List of cities fetched successfully",
                "data": cities_cache['cities']
            }

        # Fetch data from Firestore
        logger.info("Fetching events from Firestore")
        events_ref = db.collection('events')
        events_docs = events_ref.stream()
        cities = set()
        
        for doc in events_docs:
            event_data = doc.to_dict()
            logger.info(f"Fetched document: {event_data}")
            if 'Location_City' in event_data:
                cities.add(event_data['Location_City'])

        cities_list = list(cities)
        cities_cache['cities'] = cities_list  # Cache the data

        return {
            "success": True,
            "message": "List of cities fetched successfully",
            "data": cities_list
        }
    except Exception as e:
        logger.error(f'Error fetching cities: {e}')
        return {
            "success": False,
            "message": str(e),
            "data": []
        }

@app.get("/categories")
async def get_categories():
    logger.info("Received request for /categories endpoint")
    try:
        if 'categories' in categories_cache:
            logger.info("Returning cached categories data")
            return {
                "success": True,
                "message": "List of categories fetched successfully",
                "data": categories_cache['categories']
            }

        # Fetch data from Firestore
        logger.info("Fetching events from Firestore")
        events_ref = db.collection('events')
        events_docs = events_ref.stream()
        categories = set()
        
        for doc in events_docs:
            event_data = doc.to_dict()
            logger.info(f"Fetched document: {event_data}")
            if 'Category' in event_data:
                categories.add(event_data['Category'])

        categories_list = list(categories)
        categories_cache['categories'] = categories_list  # Cache the data

        return {
            "success": True,
            "message": "List of categories fetched successfully",
            "data": categories_list
        }
    except Exception as e:
        logger.error(f'Error fetching categories: {e}')
        return {
            "success": False,
            "message": str(e),
            "data": []
        }
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
