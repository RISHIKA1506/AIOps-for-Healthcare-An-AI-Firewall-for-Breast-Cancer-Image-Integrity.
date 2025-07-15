# app.py - Your Deployed Secure Prediction Web App

import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

# ================================================================
#               LOAD ALL YOUR PRE-TRAINED MODELS
#   (Streamlit will cache these so they only load once)
# ================================================================

@st.cache_resource
def load_models():
    """Loads all models and artifacts into memory."""
    try:
        scaler = joblib.load('scaler.joblib')
        pca = joblib.load('pca.joblib')
        firewall_model = joblib.load('ai_firewall_model.joblib')

        # Must include the class definition for PyTorch to load the model
        class NeuralNet14Layer(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(NeuralNet14Layer, self).__init__()
                self.layer1=nn.Linear(input_dim,512);self.layer2=nn.ReLU();self.layer3=nn.Linear(512,512);self.layer4=nn.ReLU()
                self.layer5=nn.Linear(512,512);self.layer6=nn.ReLU();self.layer7=nn.Linear(512,512);self.layer8=nn.ReLU()
                self.layer9=nn.Linear(512,512);self.layer10=nn.ReLU();self.layer11=nn.Linear(512,512);self.layer12=nn.ReLU()
                self.layer13=nn.Linear(512,128);self.layer14=nn.ReLU();self.output_layer=nn.Linear(128,output_dim)
            def forward(self,x):
                x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x);x=self.layer5(x);x=self.layer6(x)
                x=self.layer7(x);x=self.layer8(x);x=self.layer9(x);x=self.layer10(x);x=self.layer11(x);x=self.layer12(x)
                x=self.layer13(x);x=self.layer14(x);x=self.output_layer(x)
                return x

        INPUT_DIM = pca.n_components_
        OUTPUT_DIM = 3
        main_model = NeuralNet14Layer(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
        main_model.load_state_dict(torch.load('14_layer_neural_network.pth'))
        main_model.eval()
        
        return scaler, pca, firewall_model, main_model
    except FileNotFoundError as e:
        st.error(f"ERROR: A necessary model file is missing: {e}. Please run the build script first.")
        return None, None, None, None

# --- Load the models when the app starts ---
scaler, pca, firewall_model, main_model = load_models()

# ================================================================
#              HELPER FUNCTIONS FROM YOUR PIPELINE
# ================================================================
def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop).mean() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    return np.array(features)

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return np.zeros(128) if descriptors is None else descriptors.mean(axis=0)

def extract_firewall_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    mean, std_dev = cv2.meanStdDev(image)
    return np.hstack([hist, mean.flatten(), std_dev.flatten()])

CANCER_CLASSES = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# ================================================================
#              THE MAIN PREDICTION FUNCTION
# ================================================================
def secure_predict(image_array):
    """Takes a NumPy array of an image and returns a result string."""
    img_resized = cv2.resize(image_array, (128, 128))

    # === FIREWALL CHECK ===
    fw_features = extract_firewall_features(img_resized).reshape(1, -1)
    is_tampered = firewall_model.predict(fw_features)[0]

    if is_tampered == 1:
        return "Blocked", "[ALERT] AI Firewall has blocked this image. It is flagged as potentially tampered or corrupted."
    
    # === MAIN ANALYSIS ===
    st.info("AI Firewall: Image is clean. Proceeding to main cancer analysis...")
    glcm = extract_glcm_features(img_resized)
    sift = extract_sift_features(img_resized)
    combined = np.hstack([glcm, sift]).reshape(1, -1)
    
    scaled = scaler.transform(combined)
    pca_features = pca.transform(scaled)
    input_tensor = torch.tensor(pca_features, dtype=torch.float32)
    
    with torch.no_grad():
        output = main_model(input_tensor)
        _, predicted_idx = torch.max(output, 1)
        prediction = CANCER_CLASSES[predicted_idx.item()]
        
    return "Analyzed", f"Analysis complete. Final Prediction: {prediction}"

# ================================================================
#                   STREAMLIT WEB APP INTERFACE
# ================================================================

st.set_page_config(page_title="Secure Medical Image Analysis", layout="centered")

st.title("🛡️ Secure Medical Image Analysis Pipeline")
st.write(
    "This application demonstrates an end-to-end pipeline for medical image analysis, "
    "featuring an **AI Firewall** to detect and block tampered images before they "
    "reach the main diagnostic model. This is a concept inspired by Cisco's leadership in network security."
)

uploaded_file = st.file_uploader("Choose a mammogram image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an image that OpenCV can use
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Display the uploaded image
    st.image(opencv_image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Perform the prediction
    with st.spinner('Analyzing the image...'):
        status, result = secure_predict(opencv_image)

    # Display the result
    if status == "Blocked":
        st.error(result)
    else:
        st.success(result)