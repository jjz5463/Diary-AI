import openai
from google.cloud import vision
import io
import google.generativeai as genai
from diffusers import DiffusionPipeline
import torch
from diffusers.utils import export_to_video
import numpy as np
import os


# Utilize the Google Cloud Vision API to recognize text in the
# input input_images (diary input_images), https://cloud.google.com/vision.
def detect_text_in_image(image_path, credentials):

    client = vision.ImageAnnotatorClient(credentials=credentials)

    # Open the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    # Create an image object for the Vision API
    image = vision.Image(content=content)

    # Use the Vision API to detect text
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Check for errors in the response
    if response.error.message:
        raise Exception(f'{response.error.message}')

    # Return the detected text or an empty string
    return texts[0].description if texts else ''


# Utilize the PaLM 2 Bison for Text model to conduct NLP tasks such as
# text summarization and condensing on the diary text, https://ai.google.dev/palm_docs/palm.
def summarize_diary_text(text, api_key):
    # Initialize the OpenAI client
    client = openai.Client(api_key=api_key)

    # Use the client to call the chat completion API
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {"role": "user", "content": f"Summarize the following diary entry: {text}"}
        ],
        max_tokens=150,
        temperature=0.7,
        n=1  # Number of completions to generate
    )

    # Extract the summary from the response
    return response.choices[0].message.content


def break_summary_to_activities(text, api_key):
    # Initialize the OpenAI client
    client = openai.Client(api_key=api_key)

    # Use the client to call the chat completion API
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {"role": "user", "content": f"Please break the following summary into four distinct activities, "
                                        f"formatted as 'I am [activity].' Each activity should describe a unique action "
                                        f"and be less than six words: {text}. "
                                        f"Return the four activities as a list in this "
                                        f"format: [activity1, activity2, activity3, activity4], "
                                        f"without any quotation marks or extra text."}
        ],
        max_tokens=150,
        temperature=0.7,
        n=1  # Number of completions to generate
    )

    # Extract the summary from the response
    return response.choices[0].message.content


# Utilize the Gemini 1.0 Pro Vision to input an image of the diary writer,
# and output a textual description of the image,
# https://ai.google.dev/gemini-api/docs/models/gemini.
# Mock example assuming an API request to Gemini
def analyze_writer_image(image_path, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    myfile = genai.upload_file(image_path)
    result = model.generate_content(
        [myfile, "\n\n",
         "Provide a description of the people in the picture, "
         "focusing on their characteristics. Keep it under five words."]
    )
    return result.text


def generate_video(activity_list, writer_summary, fps=24):  # Lower fps
    # Load the Zeroscope video generation model
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",  # Zeroscope model from Hugging Face
        torch_dtype=torch.float16,
        cache_dir = "./zeroscope"
    )

    # Check for available device: CUDA, MPS, or CPU
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA backend.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS backend.")
    else:
        device = "cpu"
        print("CUDA and MPS not available. Falling back to CPU.")
    pipe = pipe.to(device)

    # Combine the diary text and writer description for a cohesive prompt
    prompts = []
    for activity in activity_list:
        prompt = writer_summary.strip('.').capitalize() + ' is' + activity[4:]
        prompts.append(prompt)

    # Truncate the prompt to fit the CLIP token limit
    os.makedirs("videos", exist_ok=True)
    video_paths = []
    for i, prompt in enumerate(prompts):
        video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=fps).frames
        video_path = export_to_video(np.squeeze(video_frames, axis=0), output_video_path=f'videos/video{i}.mp4')
        video_paths.append(video_path)

    return video_paths
