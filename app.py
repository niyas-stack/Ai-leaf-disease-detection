import os
import numpy as np

import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import streamlit as st
from werkzeug.utils import secure_filename

# check if a GPU is available, else use the CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model from the file path
try:
    model = torch.load("epoch-81.pt", map_location=device)
    model.eval()
except RuntimeError:
    st.error("Failed to load the model. Please check if the model file is available and try again.")
    st.stop()

classes = dict({0:'The above leaf is Cassava (Cassava Mosaic) ', 
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
                12:' The above leaf is Tomato healthy', 
                13:'The above leaf is bean angular leaf spot',
                14:'The above leaf is bean healthy', 
                15:'The above leaf is bean rust'})

# image transformation
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
    image = Image.open(img_path)
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    return pred

def main():
    st.set_page_config(page_title="Plant Disease Detection App")
    st.title("Plant Disease Detection App")
    st.write("Upload an image of a plant leaf and the app will predict whether it has a disease or not.")
    file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if file is not None:
        filename = secure_filename(file.name)
        with open(os.path.join("./tempDir", filename), "wb") as f:
            f.write(file.getbuffer())
        img_path = os.path.join("./tempDir", filename)
        st.image(img_path, use_column_width=True)
        pred = model_predict(img_path, model, transform)
        st.success(pred)
       
if __name__ == '__main__':
    main()
