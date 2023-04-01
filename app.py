import streamlit as st
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transforms for the input image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.Normalize((0.5,0.5,0.5),(1,1,1))
])

# Load the model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 16)
model.to(device)
model_path = "C:\\Users\\NOUFAL\\OneDrive\\Documents\\PROJECT\\resnet_18_epochs\\epoch-61.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define a function to make predictions
def predict(image):
    # Load and transform the input image
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Make the prediction
    output = model(image_tensor)
    index = torch.argmax(output)
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.93:
        return "Not defined"
    else:
        return classes[index]

# Define the Streamlit app
def app():
    st.title("ResNet-18 Image Classifier")
    st.write("This app uses a ResNet-18 model to classify images into 16 different classes.")
    uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make a prediction and display the result
        pred = predict(image)
        st.write("Prediction: ", pred)

# Run the app
if __name__ == '__main__':
    app()


