import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2

# Load the trained model
model = torch.load("path/to/your/model.pt")

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(1,1,1))
])

# Define the prediction function
def predict(image):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model(image_tensor)
    index = torch.argmax(output)
    pred = train_data.classes[index]
    probs, _ = torch.max(torch.softmax(output,dim=1),1)
    if probs < 0.93:
        return "not defined"
    else:
        return pred

# Define the Streamlit app
def app():
    st.set_page_config(page_title="Image Classification App", page_icon=":smiley:", layout="wide")
    st.title("Image Classification App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        pred = predict(image)
        st.write("Prediction: ", pred)

if __name__ == '__main__':
    app()




