import gradio as gr
import openai
import json
from google.oauth2 import service_account
from baseline_utils import (detect_text_in_image,
                            analyze_writer_image,
                            generate_video,
                            break_diary_to_scenes,
                            scenes_caption)
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

    # Analyze the writer's image using Gemini API
    writer_summary = analyze_writer_image(writer_image_path, gemini_api_key)

    scenes = break_diary_to_scenes(detected_text, writer_summary, openai_api_key)
    scene_list = [scene.strip() for scene in scenes.split("Scene")[1:]]
    scene_list = [scene.split(": ", 1)[1] for scene in scene_list]

    # Generate the video based on the summaries
    video_paths = generate_video(scene_list, fps=24)

    captions = scenes_caption(scene_list, openai_api_key)

    return video_paths, captions


# Define the Gradio interface
def gradio_interface(diary_image, writer_image):
    # Process the images and generate the video
    video_paths, prompts = process_images(diary_image, writer_image)

    # Return the paths and corresponding prompts
    return video_paths[0], prompts[0], video_paths[1], prompts[1], video_paths[2], prompts[2], video_paths[3], prompts[
        3]


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
            prompt_output_1 = gr.Markdown(label="Prompt for Video 1")
        with gr.Column():
            video_output_2 = gr.Video(label="Generated Video 2")
            prompt_output_2 = gr.Markdown(label="Prompt for Video 2")

    with gr.Row():
        with gr.Column():
            video_output_3 = gr.Video(label="Generated Video 3")
            prompt_output_3 = gr.Markdown(label="Prompt for Video 3")
        with gr.Column():
            video_output_4 = gr.Video(label="Generated Video 4")
            prompt_output_4 = gr.Markdown(label="Prompt for Video 4")

    # Bind the submit button click to trigger the video generation and display
    submit_button.click(fn=gradio_interface,
                        inputs=[diary_image_input, writer_image_input],
                        outputs=[video_output_1, prompt_output_1,
                                 video_output_2, prompt_output_2,
                                 video_output_3, prompt_output_3,
                                 video_output_4, prompt_output_4])

# Launch the interface
interface.launch()