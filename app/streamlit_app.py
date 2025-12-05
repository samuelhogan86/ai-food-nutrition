import streamlit as st
import torch
from PIL import Image
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)

from src.model import build_model
from src.data_loader import get_test_transforms

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names from train folder
    data_dir = os.path.join(ROOT, "data", "processed", "train")
    class_names = sorted(os.listdir(data_dir))

    num_classes = len(class_names)
    model = build_model(num_classes=num_classes)
    model_path = os.path.join(ROOT, "best_model.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, class_names, device

def preprocess_image(image: Image.Image):
    transform = get_test_transforms()
    img = transform(image).unsqueeze(0)   # (1, 3, 224, 224)
    return img

def predict(model, device, image_tensor, class_names):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class = probs.topk(1)

    predicted_label = class_names[top_class.item()]
    confidence = float(top_prob.item())

    return predicted_label, confidence

def main():
    st.title("Food Image Classifier")
    st.write("Upload an image of food, and the model will predict what dish it is.")

    model, class_names, device = load_model()

    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Convert to PIL
        image = Image.open(uploaded_file).convert("RGB")

        # Preprocess
        image_tensor = preprocess_image(image)

        # Predict
        with st.spinner("Classifying..."):
            label, conf = predict(model, device, image_tensor, class_names)

        st.subheader("Prediction")
        st.write(f"**Food Category:** `{label}`")
        st.write(f"**Confidence:** `{conf*100:.2f}%`")

if __name__ == "__main__":
    main()