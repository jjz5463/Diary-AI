import gradio as gr
import openai
import json
from google.oauth2 import service_account
from baseline_utils import detect_text_in_image, summarize_diary_text, analyze_writer_image, generate_video, break_summary_to_activities
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
    activities = break_summary_to_activities(summarized_text, openai_api_key)
    activity_list = activities.strip('[]').split(', ')

    # Analyze the writer's image using Gemini API
    writer_summary = analyze_writer_image(writer_image_path, gemini_api_key)

    # Generate the video based on the summaries
    video_paths = generate_video(activity_list, writer_summary, fps=24)

    return video_paths


# Define the Gradio interface
def gradio_interface(diary_image, writer_image):
    # Process the images and generate the video
    generated_videos = process_images(diary_image, writer_image)
    return generated_videos


# Set up the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Handwritten Diary to Video")

    with gr.Row():
        diary_image_input = gr.Image(label="Upload your handwritten diary image", type="pil")
        writer_image_input = gr.Image(label="Upload a photo of the writer", type="pil")

    submit_button = gr.Button("Generate Videos")

    with gr.Row():
        with gr.Column():
            video_output_1 = gr.Video(label="Generated Video 1")
            video_output_2 = gr.Video(label="Generated Video 2")
        with gr.Column():
            video_output_3 = gr.Video(label="Generated Video 3")
            video_output_4 = gr.Video(label="Generated Video 4")

    submit_button.click(fn=gradio_interface,
                        inputs=[diary_image_input, writer_image_input],
                        outputs=[video_output_1, video_output_2, video_output_3, video_output_4])

# Launch the interface
interface.launch()