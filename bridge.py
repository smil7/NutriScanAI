import os
import faiss
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import fasttext
import requests

# Load FastText model (you need to change this path to the actual model path)
FASTTEXT_MODEL_PATH = 'path_to_your_model/crawl-300d-2M-subword.bin'
model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Google Vision function to extract text from image
def extract_text_from_image(image_path):
    credentials = service_account.Credentials.from_service_account_file('path_to_your_service_account.json')
    client = vision.ImageAnnotatorClient(credentials=credentials)
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    
    texts = response.text_annotations
    return texts[0].description if texts else None

# Convert text to embeddings using FastText
def get_text_embedding(text):
    try:
        embedding = model.get_sentence_vector(text)
        return embedding  # Return numpy array (FAISS needs it)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# FAISS Index Setup
def setup_faiss_index(embedding_dimension):
    # L2 distance for similarity search
    index = faiss.IndexFlatL2(embedding_dimension)  # FAISS uses float32 embeddings
    return index

# Store embedding in FAISS
def store_embedding_in_faiss(index, embedding):
    # Convert the embedding to numpy array (FAISS requires float32)
    embedding_np = np.array([embedding], dtype=np.float32)
    index.add(embedding_np)  # Add embedding to FAISS index
    print("Data stored in FAISS!")

# Query FAISS and send to Gemini API
def query_faiss_and_send_to_gemini(index, query_text, k=5):
    query_embedding = get_text_embedding(query_text)
    
    if query_embedding is not None:
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        
        # Search for the top k most similar embeddings
        distances, indices = index.search(query_embedding_np, k)
        
        print(f"Found top {k} nearest neighbors with distances: {distances}")
        
        # Get the closest embedding (just for simplicity, you could handle this better)
        closest_embedding = query_embedding  # Just use query embedding for now
        
        # Send the closest embedding to Gemini API
        return send_to_gemini_api(closest_embedding)
    else:
        print("Error generating query embedding.")
        return None

# Function to send data to Gemini API
def send_to_gemini_api(embedding_data):
    # Assuming you have the Gemini API URL and API key
    GEMINI_API_URL = "https://your-gemini-api-endpoint.com/analyze"
    GEMINI_API_KEY = "your_gemini_api_key"  # Replace with your actual API key

    # Sending the embedding to Gemini API for analysis
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    payload = {"data": embedding_data.tolist()}  # Convert numpy array to list
    
    # Send POST request to Gemini API
    response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("Gemini API response:", response.json())
        return response.json()  # Return the response from Gemini for further use
    else:
        print("Error sending data to Gemini API:", response.status_code)
        return None

# Main function to orchestrate the process
def main(image_path):
    # Step 1: Extract text from image using Google Vision
    text = extract_text_from_image(image_path)
    
    if text:
        # Step 2: Convert the text to FastText embedding
        embedding = get_text_embedding(text)
        
        if embedding is not None:
            # Step 3: Set up FAISS index and store the embedding
            embedding_dimension = len(embedding)  # Typically 300 for FastText
            index = setup_faiss_index(embedding_dimension)
            store_embedding_in_faiss(index, embedding)
            
            # Step 4: Query FAISS for similar embeddings and send the result to Gemini API
            query_faiss_and_send_to_gemini(index, text)
        else:
            print("Failed to generate embedding from the text.")
    else:
        print("No text extracted from image.")

# Run the script with an example image
if __name__ == "__main__":
    image_path = 'path_to_image.jpg'  # Replace with your actual image file path
    main(image_path)
