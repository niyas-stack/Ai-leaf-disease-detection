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
        'Use of resistant variety Sripadmanaba suited for Tamil Nadu and Kerala. Mosaic tolerant varieties such as H-97 may be used to minimize economic loss of tubers. Select setts from healthy plants. Roug out and destroy infected plants in the field at early stage.Control whitefly by installing yellow sticky traps, removal of weed hosts, spray neem oil (20 ml / litre of water). Spray Dimethoate 30 EC (2 ml / litre of water) to control the vector', 'കിഴങ്ങുവർഗ്ഗങ്ങളുടെ സാമ്പത്തിക നഷ്ടം കുറയ്ക്കുന്നതിന് H-97 പോലുള്ള മൊസൈക്ക് സഹിഷ്ണുതയുള്ള ഇനങ്ങൾ ഉപയോഗിക്കാം. ആരോഗ്യമുള്ള ചെടികളിൽ നിന്ന് സെറ്റുകൾ തിരഞ്ഞെടുക്കുക. രോഗബാധയുള്ള ചെടികളെ ആദ്യഘട്ടത്തിൽ തന്നെ പറിച്ച് നശിപ്പിക്കുക. മഞ്ഞ സ്റ്റിക്കി കെണികൾ സ്ഥാപിക്കുക, കളകളെ നീക്കം ചെയ്യുക, വേപ്പെണ്ണ (20 മില്ലി / ലിറ്റർ വെള്ളത്തിൽ) തളിക്കുക എന്നിവയിലൂടെ വെള്ളീച്ചയെ നിയന്ത്രിക്കുക. വെക്‌ടറിനെ നിയന്ത്രിക്കാൻ ഡൈമെത്തോയേറ്റ് 30 ഇസി (2 മില്ലി/ലിറ്റർ വെള്ളം) തളിക്കുക.',
        'CASSAVA(MOSAIC)(ENG).ogg', 'cassava.m4a'
    ],
    'The above leaf is Cassava CB (Cassava Bacterial Blight)': [
        'Remedy for Cassava Bacterial Blight', 'കാസവ ബാക്ടീരിയൽ ബ്ലൈറ്റിന്റെ പരിഹാരം',
        'cassava.m4a', 'CASSAVA(MOSAIC)(MAL).m4a'
    ]
    # add remedies for other diseases in both English and Malayalam
}

selected_language = 'English'  # Set the default language


num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(classes))
model_path = "epoch-8.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
summary(model, input_size=(3, 224, 224))
model.eval()

# image transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def model_predict(image, model_func, transform):
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze(0)
    output = model_func(image_tensor)
    index = torch.argmax(output)
    pred = classes[index.item()]
    probs, _ = torch.max(F.softmax(output, dim=1), 1)
    if probs < 0.93:
        return "This is not trained yet ", probs
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
        st.markdown(f"<p style='color: red;'>Probability: {st.session_state['probs']:.2f}</p>", unsafe_allow_html=True)

        if not st.session_state['language_selected']:
            selected_language = st.selectbox("Select Language", ("English", "Malayalam"))
            st.session_state['selected_language'] = selected_language
            st.session_state['language_selected'] = True

        display_remedies(st.session_state['pred'])

        if st.session_state['selected_language'] == 'Malayalam':
            display_remedies_malayalam(st.session_state['pred'])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.text("AI Leaf Disease Detection - Developed by Your Name")
    st.text("Background image source: Unsplash")


if __name__ == "__main__":
    main()
