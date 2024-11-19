import openai
from google.cloud import vision
import io
import google.generativeai as genai
from diffusers import CogVideoXPipeline
import torch
from diffusers.utils import export_to_video
import os
import spaces
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

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
    color = model.generate_content([myfile,"What is the predominant color of the person in the image?"])
    description = f"""
    The main character is a cartoonish, fluffy cat with large, expressive blue eyes. 
    Its fur is predominantly {color}, with subtle shading on certain parts of its body in a slightly darker or lighter shade of {color}.
    The face is round with soft, slightly pointed ears that are highlighted with an inner coloring also in {color}.

    The most prominent feature of the cat is its extremely fluffy, oversized tail, which arcs gracefully above its body. 
    The tail fur is thick, feathery, and has a luxurious texture that stands out against the rest of the body, showcasing 
    a gradient effect from darker to lighter shades of {color} at the edges. 

    The cat’s paws are small and round, with shading in a slightly darker shade of {color}. 
    The overall look of the figure is cute, gentle, and meticulously detailed, emphasizing a soft and playful appearance.
    """
    return description


def break_diary_to_scenes(diary_text, api_key):
    # Initialize the OpenAI client
    client = openai.Client(api_key=api_key)

    example_1 = 'Main Character head bowed and determination etched on his face, sprints through the torrential downpour as lightning crackles and thunder rumbles in the distance. The relentless rain pounds the ground, creating a chaotic dance of water droplets that mirror the dramatic sky\'s anger. In the far background, the silhouette of a cozy home beckons, a faint beacon of safety and warmth amidst the fierce weather. The scene is one of perseverance and the unyielding spirit of a child braving the elements.'
    example_2 = 'Main Character, with the red dust of Mars clinging to their boots, reaches out to shake hands with an alien being, their skin a shimmering blue, under the pink-tinged sky of the fourth planet. In the background, a sleek silver rocket, a beacon of human ingenuity, stands tall, its engines powered down, as the two representatives of different worlds exchange a historic greeting amidst the desolate beauty of the Martian landscape.'
    example_3 = 'Main Character, with a serene expression, sits at the water\'s edge, a steaming cup of tea by his side. He is engrossed in his artwork, brush in hand, as he renders an oil painting on a canvas that\'s propped up against a small, weathered table. The sea breeze whispers through his silver hair, gently billowing his loose-fitting white shirt, while the salty air adds an intangible element to his masterpiece in progress. The scene is one of tranquility and inspiration, with the artist\'s canvas capturing the vibrant hues of the setting sun reflecting off the tranquil sea.'

    # Use the client to call the chat completion API
    response = client.chat.completions.create(
        model="gpt-4",  # Use GPT-4
        messages=[
            {
                "role": "user",
                "content": f"Please break the following diary into four distinct cartoon movie scenes: {diary_text}. "
                           f"Each scene should focus on one unique action and be described in vivid, animated detail. Below are some examples for the desired style: "
                           f"Example 1: {example_1}. Example 2: {example_2}. Example 3: {example_3}."
                           f"Use 'Main Character' as the placeholder for the subject in the scene. "
                           f"Each scene should be main character is doing something."
                           f"Ensure that each scene features only one action, with no combinations (e.g., avoid 'eating and teaching' in one scene). "
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

    return "\n\n".join(captions)


@spaces.GPU
def generate_video(scene_list, writer_description, fps=24):  # Lower fps

    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        torch_dtype=torch.bfloat16,
        cache_dir="./CogVideoX-5b"
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()


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

    # Truncate the prompt to fit the CLIP token limit
    os.makedirs("videos", exist_ok=True)
    video_paths = []
    for i, prompt in enumerate(scene_list):
        video = pipe(
            prompt=prompt + f'\nThe main character is described as: {writer_description}.',
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=fps,
            guidance_scale=6,
            generator=torch.Generator(device=device).manual_seed(42),
        ).frames[0]

        video_path = export_to_video(video, output_video_path=f'videos/video{i}.mp4')
        video_paths.append(video_path)

    # Concatenate the generated videos into a single video
    concatenated_video_path = "videos/combined_video.mp4"
    concatenate_videos(video_paths, concatenated_video_path, audio_path="meow-meow-meow-tiktok.mp3")
    return concatenated_video_path


def concatenate_videos(video_paths, output_path, audio_path="meow-meow-meow-tiktok.mp3"):
    # Load each video file as a VideoFileClip
    clips = [VideoFileClip(video) for video in video_paths]

    # Concatenate the clips
    final_clip = concatenate_videoclips(clips, method="compose")

    # If an audio file is provided, load it and trim if necessary
    if audio_path:
        audio = AudioFileClip(audio_path)
        # Trim the audio to match the final video's duration if audio is longer
        if audio.duration > final_clip.duration:
            audio = audio.subclip(0, final_clip.duration)
        # Set the trimmed audio to the final video clip
        final_clip = final_clip.set_audio(audio)

    # Write the concatenated video to a file
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")