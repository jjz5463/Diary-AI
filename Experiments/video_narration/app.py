import gradio as gr
import openai
import json
from google.oauth2 import service_account
from baseline_utils import (detect_text_in_image,
                            analyze_writer_image,
                            generate_video,
                            break_diary_to_scenes,
                            scenes_caption,
                            summarizer_for_audio,
                            narration_generate)
import os

# Load secrets from Hugging Face Spaces environment
openai_api_key = os.getenv("OPENAI_API_KEY")
google_service_account_info = json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT"))
gemini_api_key = os.getenv("GEMINI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

# Initialize OpenAI
openai.api_key = openai_api_key


# Function to get Google credentials
def get_google_credentials():
    return service_account.Credentials.from_service_account_info(google_service_account_info)


def process_images(diary_image, writer_image, audio_option):
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

    scenes = break_diary_to_scenes(detected_text, openai_api_key)
    scene_list = [scene.strip() for scene in scenes.split("Scene")[1:]]
    scene_list = [scene.split(": ", 1)[1] for scene in scene_list]

    # Generate the narration audio which is less than 10 second
    # This will create a mp3 file for narration
    narration_summarize = summarizer_for_audio(detected_text)
    narration_generate(narration_summarize, eleven_api_key)
    # Generate the video based on the summaries
    video_path = generate_video(scene_list, writer_summary, audio_option, fps=24)

    caption = scenes_caption(scene_list, openai_api_key)

    return video_path, caption


# Define the Gradio interface
def gradio_interface(diary_image, writer_image, audio_option):
    # Process the images and generate the video
    video_paths, prompts = process_images(diary_image, writer_image, audio_option)

    # Return the paths and corresponding prompts
    return video_paths, prompts


# Set up the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown("# Handwritten Diary to Video")

    with gr.Row():
        # Left column for user inputs
        with gr.Column():
            diary_image_input = gr.Image(label="Upload your handwritten diary image", type="pil")
            writer_image_input = gr.Image(label="Upload a photo of the writer", type="pil")
            # Add a radio button for selecting audio options
            audio_option = gr.Radio(
                ["Narration", "Meow"],
                label="Choose Audio Option",
                value="Narration"  # Default selection
            )
            submit_button = gr.Button("Generate Video")

        # Right column for generated video and caption
        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            caption_output = gr.Markdown(label="Scene Caption")

    # Bind the submit button click to trigger the video generation and display
    submit_button.click(
        fn=gradio_interface,
        inputs=[diary_image_input, writer_image_input, audio_option],
        outputs=[video_output, caption_output]
    )

# Launch the interface
interface.launch()