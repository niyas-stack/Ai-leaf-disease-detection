import os
import numpy as np
from torchsummary import summary
import torch
import PIL
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import streamlit as st
import base64

# Load the model
model = torchvision.models.resnet18(pretrained=True)
classes = dict({0:'The above leaf is Cassava (Cassava Mosaic) ', 
                1:'The above leaf is Cassava CB (Cassava Bacterial Blight)', 
                2:'The above leaf is Cassava Healthy leaf', 
                3:'This is not trained yet',
                4:'The above leaf is Tomato Bacterial spot', 
                5:'The above leaf is Tomato early blight',
                6:'The above leaf is Tomato Late blight',
                7:'The above leaf is Tomato Leaf Mold', 
                8:'The above leaf is Tomato Septoria leaf spot',
                9:'The above leaf is Tomato Spider mites Two-spotted spider mite', 
                10:'The above leaf is Tomato Target Spot',
                11:'The above leaf is Tomato Yellow Leaf Curl Virus', 
                12:'The above leaf is Tomato mosaic virus', 
                13:' The above leaf is Tomato healthy', 
                14:'The above leaf is bean angular leaf spot',
                15:'The above leaf is bean healthy', 
                16:'The above leaf is bean rust'})
remedies = {
    'The above leaf is Cassava (Cassava Mosaic)': [
        'Remedy for Cassava Mosaic', 'കാസവ മോസായികയുടെ പരിഹാരം',
        '96_Songs_The_Life_of_Ram_Video_Song_Vijay_Sethupathi,_Trisha_Govind.mp3', 'cassava.m4a'
    ],
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': [
        'Remedy for Cassava Bacterial Blight', 'കാസവ ബാക്ടീരിയൽ ബ്ലൈറ്റിന്റെ പരിഹാരം',
        'cassava.m4a', 'cassava.m4a'
    ]
    # add remedies for other diseases in both English and Malayalam
}

selected_language = 'English'  # Set the default language


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-90.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
summary(model, input_size=(3, 224, 224))
model.eval()

#image transformation
transform=transforms.Compose([
transforms.ToTensor(),
    transforms.Resize((224,224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
def model_predict(img_path,model_func,transform):
    image=Image.open(img_path)
    image_tensor=transform(image).float()
    image_tensor=image_tensor.unsqueeze(0)
    image_tensor=image_tensor.to(device)
    output=model_func(image_tensor)
    index=torch.argmax(output)
    print(index)
    pred=classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.93:
        return "not defined", probs
    else:
        return pred, probs



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def display_remedies(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy:</p>", unsafe_allow_html=True)
        if selected_language == 'English':
            audio_file = remedy[2]
        else:
            audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        if selected_language == 'English':
            st.success(f" {remedy[0]}")
        else:
            st.success(f" {remedy[1]}")

def display_remedies_malayalam(pred):
    remedy = remedies.get(pred)
    if remedy:
        st.markdown("<p style='color:red;'>Remedy (Malayalam):</p>", unsafe_allow_html=True)
        audio_file = remedy[3]
        with open(audio_file, 'rb') as audio:
            st.audio(audio.read(), format='audio/mp3')
        st.success(f" {remedy[1]}")

def init_session_state():
    if 'pred' not in st.session_state:
        st.session_state['pred'] = None
    if 'probs' not in st.session_state:
        st.session_state['probs'] = None
    if 'selected_language' not in st.session_state:
        st.session_state['selected_language'] = 'English'
    if 'language_selected' not in st.session_state:
        st.session_state['language_selected'] = False

def clear_session_state():
    st.session_state['pred'] = None
    st.session_state['probs'] = None
    st.session_state['language_selected'] = False

def main():
    init_session_state()

    st.set_page_config(page_title="AI Leaf Disease Detection", page_icon=":leaves:")
    st.markdown("<h1 style='color: green;'>AI Leaf Disease Detection</h1>", unsafe_allow_html=True)
    add_bg_from_local('background app2a.jpg')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        clear_session_state()  # Clear session state when a new file is uploaded

        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write("")

        if st.button("Classify", key="classify_btn"):
            pred, probs = model_predict(image, model, transform)
            st.session_state['pred'] = pred
            st.session_state['probs'] = probs.item()
            st.session_state['language_selected'] = False

    if st.session_state['pred'] is not None:
        st.markdown(f"<p style='color: red;'>Prediction: {st.session_state['pred']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: red;'>Probability: {st.session_state['probs']}</p>", unsafe_allow_html=True)

        if st.session_state['pred'] != "not defined":
            selected_language = st.selectbox("Select Language", ['English', 'Malayalam'], index=0, key="language_select")
            st.session_state['selected_language'] = selected_language

            if st.session_state['selected_language'] == 'Malayalam':
                display_remedies_malayalam(st.session_state['pred'])
            else:
                display_remedies(st.session_state['pred'])

if __name__ == "__main__":
    main()
