import gradio as gr
import openai
import json
from PIL import Image
from google.oauth2 import service_account
from baseline_utils import detect_text_in_image, summarize_diary_text, analyze_writer_image, generate_comic_book
import glob
import os
from keys.keys import *

# Load secrets from the environment or other sources (adjust as needed)
openai_api_key = open_ai_keys
with open('keys/service_account_credentials.json') as f:
    google_service_account_info = json.load(f)
gemini_api_key = gemini_keys

# Initialize OpenAI
openai.api_key = openai_api_key


# Function to get Google credentials
def get_google_credentials():
    return service_account.Credentials.from_service_account_info(google_service_account_info)


def process_images(diary_image, writer_image):
    # Save the file-like objects as image files
    diary_image_path = "temp_upload_images/temp_diary_image.png"
    writer_image_path = "temp_upload_images/temp_writer_image.png"
    os.makedirs("temp_upload_images", exist_ok=True)
    diary_image.save(diary_image_path)
    writer_image.save(writer_image_path)

    # Detect text from the diary image
    google_credentials = get_google_credentials()
    detected_text = detect_text_in_image(diary_image_path, google_credentials)
    summarized_text = summarize_diary_text(detected_text, openai_api_key)

    # Analyze the writer's image using Gemini API
    writer_summary = analyze_writer_image(writer_image_path, gemini_api_key)

    # Generate the comic book based on the summaries
    generate_comic_book(summarized_text, writer_summary, num_pages=4)

    # Assuming generated images are saved as 'comic_book/page_1.png', 'comic_book/page_2.png', etc.
    image_files = sorted(glob.glob("comic_book/page_*.png"))  # Find all the generated comic book pages

    return image_files


# Define the Gradio interface
def gradio_interface(diary_image, writer_image):
    # Process the images and generate comic book pages
    generated_images = process_images(diary_image, writer_image)

    # Load the images and return them
    images = [Image.open(img) for img in generated_images]
    return images


# Set up the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(label="Upload your handwritten diary image", type="pil"),
        gr.Image(label="Upload a photo of the writer", type="pil"),
    ],
    outputs=gr.Gallery(label="Generated Comic Book Pages"),
    title="Handwritten Diary to Comic Book"
)

# Launch the interface
interface.launch()