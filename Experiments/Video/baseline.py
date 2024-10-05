
import openai
import json
from google.oauth2 import service_account
from baseline_utils import detect_text_in_image, analyze_writer_image, break_diary_to_scenes, scenes_caption
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



diary_image_path = "temp_upload_images/temp_diary_image.png"
writer_image_path = "temp_upload_images/temp_writer_image.png"

# Detect text from the diary image
google_credentials = get_google_credentials()
detected_text = detect_text_in_image(diary_image_path, google_credentials)

# Analyze the writer's image using Gemini API
writer_summary = analyze_writer_image(writer_image_path, gemini_api_key)

scenes = break_diary_to_scenes(detected_text, writer_summary, openai_api_key)
scene_list = [scene.strip() for scene in scenes.split("Scene")[1:]]
scene_list = [scene.split(": ", 1)[1] for scene in scene_list]

captions = scenes_caption(scene_list, openai_api_key)

print(captions)
