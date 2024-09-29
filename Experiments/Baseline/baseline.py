from baseline_utils import *
from keys.keys import *

diary_image_path = "input_images/test_sample.jpeg"
writer_image_path = "input_images/writer.jpeg"
credentials_path = "keys/service_account_credentials.json"

# Detect text from the image using the provided credentials
detected_text = detect_text_in_image(diary_image_path, credentials_path)
diary_summary = summarize_diary_text(detected_text, open_ai_keys)
writer_summary = analyze_writer_image(writer_image_path, gemini_keys)
generate_comic_book(diary_summary, writer_summary)