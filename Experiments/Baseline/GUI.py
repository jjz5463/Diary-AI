import streamlit as st
from PIL import Image

# You can create a web or mobile-based GUI so that users can experience your solution. Suggested libraries include https://www.gradio.app/ or https://streamlit.io/.
st.title('Handwritten Diary to Cartoon Book')
uploaded_diary = st.file_uploader("Upload your diary image", type=["png", "jpg", "jpeg"])
uploaded_writer_image = st.file_uploader("Upload your photo", type=["png", "jpg", "jpeg"])

if uploaded_diary and uploaded_writer_image:
    st.write("Analyzing your diary...")

    diary_text = detect_text_in_image(uploaded_diary)
    summarized_text = summarize_diary_text(diary_text)

    st.write(f"Summarized Diary Text: {summarized_text}")

    writer_description = analyze_writer_image(uploaded_writer_image)
    st.write(f"Diary Writer Description: {writer_description}")

    # Generate cartoon image
    prompt = f"{summarized_text}, featuring a person who {writer_description}"
    generated_image = generate_image(prompt)

    st.image(generated_image, caption="Generated Cartoon Image")