import os
import faiss
import numpy as np
# import fasttext
import requests
from google.cloud import vision
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
# from google.oauth2 import service_account

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/nitro/OneDrive - University of Sharjah/Desktop/me/unies-shit/mcmaster/hackathon/NutriScanAI/.nutri-scan-api-671d1d976916.json"

# Load FastText model (you need to change this path to the actual model path)
# FASTTEXT_MODEL_PATH = 'path_to_your_model/crawl-300d-2M-subword.bin'
# model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Vertex AI embeddings
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

def map_role(role):
    if role == "model":
        return "assistant"
    else:
        return role

def extract_text_from_image(images_list):
    """Detects text in multiple images and returns a list of extracted text."""
    
    client = vision.ImageAnnotatorClient()
    extracted_texts = []

    for image_path in images_list:
        with open(image_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if texts:
            txt = str(texts[0]).replace('locale: "en"', "")
            start = txt.find("bounding_poly")
            description = txt[:start] if start != -1 else txt
            extracted_texts.append(description)
        else:
            extracted_texts.append("No text detected")

        # Check for API errors
        if response.error.message:
            raise Exception(
                "{}\nFor more info on error messages, check: "
                "https://cloud.google.com/apis/design/errors".format(response.error.message)
            )
    print("From utils file: ")
    print(extracted_texts)  # List of extracted texts from all images
    return extracted_texts

# Convert text to embeddings using Vertex AI Embeddings
def embed_text(text_list) -> list[list[float]]:
    """Embeds texts with a pre-trained, foundational model.

    Returns:
        A list of lists containing the embedding vectors for each input text
    """

    # A list of texts to be embedded.
    #texts = ["banana muffins? ", "banana bread? banana muffins?"]
    #texts = extract_text_from_image(images_list)
    # The dimensionality of the output embeddings.
    dimensionality = 256
    # The task type for embedding. Check the available tasks in the model's documentation.
    task = "RETRIEVAL_DOCUMENT"

    inputs = [TextEmbeddingInput(text, task) for text in text_list]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)

    print(embeddings)
    # Example response:
    # [[0.006135190837085247, -0.01462465338408947, 0.004978656303137541, ...], [0.1234434666, ...]],
    return [embedding.values for embedding in embeddings]

# Convert text to embeddings using FastText
# def get_text_embedding(text):
#     try:
#         embedding = model.get_sentence_vector(text)
#         return embedding  # Return numpy array (FAISS needs it)
#     except Exception as e:
#         print(f"Error generating embedding: {e}")
#         return None

# FAISS Index Setup
def setup_faiss_index(embedding_dimension):
    # L2 distance for similarity search
    index = faiss.IndexFlatL2(embedding_dimension)  # FAISS uses float32 embeddings
    return index

# Store embedding in FAISS
def store_embedding_in_faiss(embedding_dim, embedding=256):
    # Convert the embedding to numpy array (FAISS requires float32)
    index = setup_faiss_index(embedding)
    embedding_np = np.array([embedding_dim], dtype=np.float32)
    faiss.normalize_L2(embedding_np)
    index.add(embedding_np)  # Add embedding to FAISS index
    print("Data stored in FAISS!")

# Query FAISS and send to Gemini API
def query_faiss_and_send_to_gemini(index, query_text, k=5):
    query_embedding = model.get_embeddings(query_text)
    
    if query_embedding is not None:
        query_embedding_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding_np)
        # Search for the top k most similar embeddings
        distances, indices = index.search(query_embedding_np, k)
        
        print(f"Found top {k} nearest neighbors with distances: {distances}")
        
        # Get the closest embedding (just for simplicity, you could handle this better)
        closest_embedding = indices[0][0] # query_embedding  # Just use query embedding for now
        
        # Send the closest embedding to Gemini API
        return closest_embedding
    else:
        print("Error generating query embedding.")
        return None

# Function to send data to Gemini API
# def send_to_gemini_api(embedding_data):
#     gemini = genai.GenerativeModel(
#         model_name="gemini-2.0-flash-exp",
#         generation_config=generation_config,
#         system_instruction="""You are an expert at analyzing nutrition facts from products and providing a recommendations for the user.
#             Your task is to engage in conversations about nutritions, dietry and symptoms and answer questions.
#             Use the nutrition facts that are stored in the vector database as well as the prompt the user has prompted
#             to provide a specific recommendation for their usecase. If there was a prompt that the user entered is outside
#             your expertise then inform the user that that's not your expertise""",
#     )

#     # Sending the embedding to Gemini API for analysis
#     headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
#     payload = {"data": embedding_data.tolist()}  # Convert numpy array to list
    
#     # Send POST request to Gemini API
#     response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
    
#     if response.status_code == 200:
#         print("Gemini API response:", response.json())
#         return response.json()  # Return the response from Gemini for further use
#     else:
#         print("Error sending data to Gemini API:", response.status_code)
#         return None

# # Main function to orchestrate the process
# def main(image_path):
#     # Step 1: Extract text from image using Google Vision
#     text = extract_text_from_image(image_path)
    
#     if text:
#         # Step 2: Convert the text to FastText embedding
#         embedding = get_text_embedding(text)
        
#         if embedding is not None:
#             # Step 3: Set up FAISS index and store the embedding
#             embedding_dimension = len(embedding)  # Typically 300 for FastText
#             index = setup_faiss_index(embedding_dimension)
#             store_embedding_in_faiss(index, embedding)
            
#             # Step 4: Query FAISS for similar embeddings and send the result to Gemini API
#             query_faiss_and_send_to_gemini(index, text)
#         else:
#             print("Failed to generate embedding from the text.")
#     else:
#         print("No text extracted from image.")

# # Run the script with an example image
# if __name__ == "__main__":
#     image_path = 'path_to_image.jpg'  # Replace with your actual image file path
#     main(image_path)
