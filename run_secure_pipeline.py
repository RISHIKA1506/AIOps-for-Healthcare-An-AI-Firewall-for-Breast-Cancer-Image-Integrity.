# run_secure_pipeline.py

# ================================================================
#                       1. IMPORTS & SETUP
# ================================================================
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
from skimage.feature import graycomatrix, graycoprops

# Must include the class definition here so torch knows how to load the model
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

# ================================================================
#              2. LOAD ALL PRE-TRAINED MODELS & ARTIFACTS
# ================================================================
print("--- Loading all pre-trained models and artifacts ---")
try:
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca.joblib')
    firewall_model = joblib.load('ai_firewall_model.joblib')

    INPUT_DIM = pca.n_components_
    OUTPUT_DIM = 3 # Benign, Malignant, Normal
    main_model = NeuralNet14Layer(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)
    main_model.load_state_dict(torch.load('14_layer_neural_network.pth'))
    main_model.eval()
    print("All models loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: A necessary model file is missing: {e}")
    print("Please run 'build_all_models.py' and your training scripts first.")
    exit()

# ================================================================
#              3. DEFINE THE SECURE PIPELINE LOGIC
# ================================================================
# --- Helper functions (copied from training scripts) ---
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

# --- The Main Secure Prediction Function ---
CANCER_CLASSES = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

def secure_predict(image_path):
    print(f"\nAnalyzing image: {image_path}")
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Could not load image. Check the file path."
        img_resized = cv2.resize(img, (128, 128))
    except Exception as e:
        return f"Error processing image file: {e}"

    # === THIS IS THE FIREWALL CHECK ===
    fw_features = extract_firewall_features(img_resized).reshape(1, -1)
    is_tampered = firewall_model.predict(fw_features)[0]

    if is_tampered == 1:
        # If the firewall blocks it, the function stops here.
        return "[ALERT] AI Firewall has blocked this image. It is flagged as potentially tampered or corrupted."
    
    # If the image is clean, we proceed.
    print("AI Firewall: Image is clean. Proceeding to main cancer analysis...")
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
        
    return f"Analysis complete. Final Prediction: {prediction}"

# ================================================================
#                         4. DEMONSTRATION
# ================================================================
if __name__ == "__main__":
    # --- IMPORTANT: UPDATE THESE TWO PATHS FOR YOUR DEMO ---
    
    # 1. Find a path to any original, clean image from your dataset
    clean_image_path = r"C:\Users\Rishika\OneDrive\Desktop\normal\normal (1).png"
    
    # 2. Create a tampered image:
    #    - Go to your 'normal' folder.
    #    - Copy an image. Paste it somewhere else (like your Desktop).
    #    - Rename it to 'tampered_image.png'.
    #    - Open it in MS Paint, draw a red line on it, and save it.
    #    - Put the path to that new image here.
    tampered_image_path = r"C:\Users\Rishika\OneDrive\Desktop\tampered_image.png"
    
    print("\n" + "="*50)
    print("               DEMONSTRATION START")
    print("="*50)
    
    # --- Test Case 1: Clean Image ---
    result_clean = secure_predict(clean_image_path)
    print(f"\n--> FINAL RESULT FOR CLEAN IMAGE: {result_clean}")
    
    print("\n" + "-"*50 + "\n")

    # --- Test Case 2: Tampered Image ---
    result_tampered = secure_predict(tampered_image_path)
    print(f"\n--> FINAL RESULT FOR TAMPERED IMAGE: {result_tampered}")
    
    print("\n" + "="*50)
    print("                DEMONSTRATION END")
    print("="*50)