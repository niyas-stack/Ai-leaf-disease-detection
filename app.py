import os
import numpy as np
import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from werkzeug.utils import secure_filename
import streamlit as st

# Import the model
model_path = "epoch-81.pt"
model = torch.load(model_path, map_location=torch.device('cpu'))

# Define the classes
classes = {
    0:'The above leaf is Cassava (Cassava Mosaic)',
    1:'The above leaf is Cassava CB (Cassava Bacterial Blight)',
    2:'The above leaf is Cassava Healthy leaf',
    3:'The above leaf is Tomato Bacterial spot',
    4:'The above leaf is Tomato early blight',
    5:'The above leaf is Tomato Late blight',
    6:'The above leaf is Tomato Leaf Mold',
    7:'The above leaf is Tomato Septoria leaf spot',
    8:'The above leaf is Tomato Spider mites Two-spotted spider mite',
    9:'The above leaf is Tomato Target Spot',
    10:'The above leaf is Tomato Yellow Leaf Curl Virus',
    11:'The above leaf is Tomato mosaic virus',
    12:'The above leaf is Tomato healthy',
    13:'The above leaf is bean angular leaf spot',
    14:'The above leaf is bean healthy',
    15:'The above leaf is bean rust'
}

# Define the image transformation
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

def model_predict(img_path, model_func, transform):
    image=Image.open(img_path)
    image_tensor=transform(image).float()
    image_tensor=image_tensor.unsqueeze(0)
    output=model_func(image_tensor)
    index=torch.argmax(output)
    pred=classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    return pred

def main():
    st.title("Plant Disease Detection App")
    st.text("Upload a picture of a plant leaf and get a prediction")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save image
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        img_path = os.path.join("uploads", uploaded_file.name)

        # Make prediction
        prediction = model_predict(img_path=img_path, model_func=model, transform=transform)

        # Show result
        st.image(uploaded_file, caption=prediction, use_column_width=True)

if __name__ == "__main__":
    main()
