
import openai
import json
from google.oauth2 import service_account
from baseline_utils import detect_text_in_image, summarize_diary_text, generate_video
from keys.keys import *

# Load secrets from the environment or other sources (adjust as needed)
openai_api_key = open_ai_keys
with open('../Video/keys/service_account_credentials.json') as f:
    google_service_account_info = json.load(f)
gemini_api_key = gemini_keys

# Initialize OpenAI
openai.api_key = openai_api_key


# Function to get Google credentials
def get_google_credentials():
    return service_account.Credentials.from_service_account_info(google_service_account_info)



diary_image_path = "../Video/temp_upload_images/temp_diary_image.png"
writer_image_path = "https://jiachengzhu.com/assets/img/me1.jpg"

# Detect text from the diary image
google_credentials = get_google_credentials()
detected_text = detect_text_in_image(diary_image_path, google_credentials)
diary_summary = summarize_diary_text(detected_text, open_ai_keys)
print(diary_summary)
# Analyze the writer's image using Gemini API
#writer_summary = analyze_writer_image(writer_image_path, gemini_api_key)
#generate_video(runway_keys, writer_image_path, diary_summary)



