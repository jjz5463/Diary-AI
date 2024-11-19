import openai
from google.cloud import vision
import io
import google.generativeai as genai
from diffusers import CogVideoXPipeline
import torch
from diffusers.utils import export_to_video
import numpy as np
import os
import spaces
import time
from runwayml import RunwayML

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


# Utilize the Gemini 1.0 Pro Vision to input an image of the diary writer,
# and output a textual description of the image,
# https://ai.google.dev/gemini-api/docs/models/gemini.
# Mock example assuming an API request to Gemini
def analyze_writer_image(image_path, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    myfile = genai.upload_file(image_path)
    result = model.generate_content(
        [myfile,
         "Provide a detailed description of the people in the picture by completely transforming them into a spiritual or symbolic animal. "
         "Do not describe them as humans; instead, fully embody them as an animal. Describe their appearance, personality, and mood "
         "as if they were this animal. For example, if someone has calm and gentle features, you might describe them as a panda. "
         "Ensure that the entire description is centered on their transformation into this animal, using vivid and expressive language, "
         "but without mentioning them as a person at all. Keep the description concise, without line breaks or unnecessary characters."]
    )
    return result.text


def break_diary_to_scenes(diary_text, writer_description, api_key):
    # Initialize the OpenAI client
    client = openai.Client(api_key=api_key)


    example_1 = 'A garden comes to life as a kaleidoscope of butterflies flutters amidst the blossoms, their delicate wings casting shadows on the petals below. In the background, a grand fountain cascades water with a gentle splendor, its rhythmic sound providing a soothing backdrop. Beneath the cool shade of a mature tree, a solitary wooden chair invites solitude and reflection, its smooth surface worn by the touch of countless visitors seeking a moment of tranquility in nature\'s embrace.'
    example_2 = 'A small boy, head bowed and determination etched on his face, sprints through the torrential downpour as lightning crackles and thunder rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the dramatic sky\'s anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements.'
    example_3 = 'A suited astronaut, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.'
    example_4 = 'An elderly gentleman, with a serene expression, sits at the water\'s edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that\'s propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist\'s canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea.'
    example_5 = 'A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog\'s energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer.'

    # Use the client to call the chat completion API
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {
                "role": "user",
                "content": f"Please break the following diary into four distinct cartoon movie scenes: {diary_text}. Each scene should focus on one unique action and be described in vivid, animated detail. Below are some examples for the desired style: "
                           f"Example 1: {example_1}. Example 2: {example_2}. Example 3: {example_3}. Example 4: {example_4}. Example 5: {example_5}. "
                           f"Ensure that each scene features only one action, with no combinations (e.g., avoid 'eating and teaching' in one scene). The main character is described as: {writer_description}. "
                           f"Please use expressive, cinematic language to bring the cartoon scene to life, focusing on the character’s actions, expressions, and environment. "
                           f"Return the output as a list in this format: Scene 1: , Scene 2: , Scene 3: , Scene 4: , without any quotation marks or line breaks."
            }
        ],
        max_tokens=1000,
        temperature=1,
        n=1  # Number of completions to generate
    )

    # Extract the summary from the response
    return response.choices[0].message.content


def scenes_caption(scenes, api_key):
    # Initialize the OpenAI client
    client = openai.Client(api_key=api_key)

    captions = []

    for scene in scenes:
        # Use OpenAI's GPT API to generate a caption for each scene
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Given the scene: {scene}, "
                               f"turn this scene into a simple caption starting with I am doing something."
                               f"Be concise, keeping it under 10 words. Return without any quotation marks."
                }
            ],
            max_tokens=50,  # Limit to a reasonable number of tokens for short captions
            temperature=0.7,  # Adjust creativity level as needed
            n=1
        )
        # Append the generated caption to the list
        captions.append(response.choices[0].message.content)

    return captions


def summarize_diary_text(text, api_key):
    # Initialize the OpenAI client
    client = openai.Client(api_key=api_key)

    # Use the client to call the chat completion API
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following diary entry under 450 characters: {text}"}
        ],
        max_tokens=150,
        temperature=0.7,
        n=1  # Number of completions to generate
    )

    # Extract the summary from the response
    return response.choices[0].message.content


@spaces.GPU
def generate_video(api_key, writer_image_path, diary_text):  # Lower fps
    # Load the Zeroscope video generation model
    # pipe = DiffusionPipeline.from_pretrained(
    #     "cerspense/zeroscope_v2_576w",  # Zeroscope model from Hugging Face
    #     torch_dtype=torch.float16,
    #     cache_dir = "./zeroscope"
    # )

    client = RunwayML(api_key=api_key)

    # Create a new image-to-video task using the "gen3a_turbo" model
    task = client.image_to_video.create(
        model='gen3a_turbo',
        # Point this at your own image file
        prompt_image=writer_image_path,
        prompt_text=diary_text
    )
    task_id = task.id

    # Poll the task until it's complete
    time.sleep(10)  # Wait for a second before polling
    task = client.tasks.retrieve(task_id)
    while task.status not in ['SUCCEEDED', 'FAILED']:
        time.sleep(10)  # Wait for ten seconds before polling
        task = client.tasks.retrieve(task_id)

    print('Task complete:', task)

    #return video_paths
