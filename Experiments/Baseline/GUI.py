import streamlit as st
import openai
import json
from PIL import Image
from google.oauth2 import service_account
from baseline_utils import detect_text_in_image, summarize_diary_text, analyze_writer_image, generate_comic_book
import glob
import os

# Load secrets
openai_api_key = st.secrets["general"]["openai_api_key"]
google_service_account_info = json.loads(st.secrets["general"]["google_service_account"])
gemini_api_key = st.secrets["general"]["gemini_api_key"]

# Initialize OpenAI
openai.api_key = openai_api_key

# Function to get Google credentials
def get_google_credentials():
    return service_account.Credentials.from_service_account_info(google_service_account_info)

st.title('Handwritten Diary to Comic Book')
uploaded_diary = st.file_uploader("Upload your handwritten diary image", type=["png", "jpg", "jpeg"])
uploaded_writer_image = st.file_uploader("Upload a photo of the writer", type=["png", "jpg", "jpeg"])

if uploaded_diary and uploaded_writer_image:
    st.write("Analyzing your diary and writer...")

    # Read the uploaded images using file-like objects
    diary_image = Image.open(uploaded_diary)
    writer_image = Image.open(uploaded_writer_image)

    # Save the file-like objects as image files (optional if needed)
    diary_image_path = "temp_upload_images/temp_diary_image.png"
    writer_image_path = "temp_upload_images/temp_writer_image.png"
    os.makedirs("temp_upload_images", exist_ok=True)
    diary_image.save(diary_image_path)
    writer_image.save(writer_image_path)

    # Detect text from the diary image
    google_credentials = get_google_credentials()
    detected_text = detect_text_in_image(diary_image_path, google_credentials)
    summarized_text = summarize_diary_text(detected_text, openai_api_key)
    st.write(f"Summarized Diary Text: {summarized_text}")

    # Analyze the writer's image using Gemini API
    writer_summary = analyze_writer_image(writer_image_path, gemini_api_key)
    st.write(f"Writer Description: {writer_summary}")

    # Generate the comic book based on the summaries
    st.write("Generating comic book images...")
    generate_comic_book(summarized_text, writer_summary, num_pages=4)

    st.write("Comic book generated successfully!")

    # Assuming generated images are saved as 'comic_book/page_1.png', 'comic_book/page_2.png', etc.
    image_files = sorted(glob.glob("comic_book/page_*.png"))  # Find all the generated comic book pages

    # Display images in 2 columns
    cols = st.columns(2)  # Create two columns for the images

    for i, image_file in enumerate(image_files):
        with cols[i % 2]:  # Alternate between the two columns
            # Display each comic book page in the respective column
            st.image(image_file, caption=image_file.split('/')[-1], use_column_width=True)