import os
import numpy as np
import pandas as pd
from google.cloud import vision
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/nitro/OneDrive - University of Sharjah/Desktop/me/unies-shit/mcmaster/hackathon/NutriScanAI/.nutri-scan-api-671d1d976916.json"

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
