import openai
from google.cloud import vision
from google.oauth2 import service_account
import io
import google.generativeai as genai
from diffusers import AutoPipelineForText2Image
import torch


# Utilize the Google Cloud Vision API to recognize text in the
# input images (diary images), https://cloud.google.com/vision.
def detect_text_in_image(image_path, credentials_path):
    # Load the service account key from the credentials JSON file
    credentials = service_account.Credentials.from_service_account_file(credentials_path)

    # Create a Vision API client using the credentials
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
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following diary entry: {text}"}
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
        [myfile, "\n\n", "Can you give a textual description of the image?"]
    )
    return result.text


# Now that you have text from the diary and text describing the diary writer,
# you can utilize the SDXL-Turbo stable diffusion model to generate
# images https://huggingface.co/stabilityai/sdxl-turbo.
# You can try to output several images for a diary entry. Analyze how accurate the results,
# and think about what could be improved.
def generate_image(diary_text, writer_description):
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir="./SDXL-Turbo")

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

    # Move the model to the selected device
    pipe = pipe.to(device)

    # Generate the image with a simple prompt
    prompt = f'Writer Description: {writer_description} \n\n Diary: {diary_text}'
    print(prompt)
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    # Save the generated image
    image.save("generated_image.png")

