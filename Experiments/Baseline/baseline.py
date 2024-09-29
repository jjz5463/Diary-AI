import json
from google.oauth2 import service_account
from baseline_utils import *
from keys.keys import *

diary_image_path = "input_images/test_sample.jpeg"
writer_image_path = "input_images/writer.jpeg"
credentials_path = "keys/service_account_credentials.json"
with open(credentials_path) as f:
    google_service_account_info = json.load(f)


def get_google_credentials():
    return service_account.Credentials.from_service_account_info(google_service_account_info)


google_credentials = get_google_credentials()

# Detect text from the image using the provided credentials
detected_text = detect_text_in_image(diary_image_path, google_credentials)
diary_summary = summarize_diary_text(detected_text, open_ai_keys)
writer_summary = analyze_writer_image(writer_image_path, gemini_keys)
generate_comic_book(diary_summary, writer_summary)