import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the BLIP model and processor
@st.cache_resource()
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

processor, model = load_blip_model()

# Streamlit UI
st.title("📸 Image Capture & Upload for Captioning")
st.write("Capture an image from your webcam or upload one, then ask multiple questions about it.")

# Options for image input
image_source = st.radio("Choose an image source:", ("Capture from Webcam", "Upload from Files"))

if image_source == "Capture from Webcam":
    image_file = st.camera_input("Take a picture")
elif image_source == "Upload from Files":
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Convert image to PIL format
    raw_image = Image.open(image_file)
    st.image(raw_image, caption="Selected Image", use_column_width=True)

    # Input for multiple questions
    st.write("### Ask multiple questions about the image:")
    num_questions = st.number_input("How many questions?", min_value=1, max_value=10, value=1, step=1)

    questions = []
    for i in range(num_questions):
        question = st.text_input(f"Question {i+1}:", key=f"q{i}")
        questions.append(question)

    # Process questions when the user clicks the button
    if st.button("Get Answers"):
        st.write("🔄 Processing your questions...")

        for i, question in enumerate(questions):
            if question.strip():  # Only process non-empty questions
                inputs = processor(raw_image, question, return_tensors="pt")

                with torch.no_grad():
                    out = model.generate(**inputs)

                # Decode the answer
                answer = processor.tokenizer.decode(out[0], skip_special_tokens=True)
                st.success(f"📝 **Q{i+1}:** {question}\n**Answer:** {answer}")

    # Clear Button to reset input fields
    if st.button("Clear"):
        st.experimental_rerun()
