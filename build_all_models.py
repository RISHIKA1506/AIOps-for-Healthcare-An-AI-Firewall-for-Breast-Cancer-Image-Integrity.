# build_all_models.py
# A single script to create and save scaler.joblib, pca.joblib, and ai_firewall_model.joblib

import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("--- MASTER BUILD SCRIPT STARTED ---")

# ================================================================
#               1. DEFINE ALL HELPER FUNCTIONS
# ================================================================
def load_images_from_folder(folder_path, limit=400):
    images = []
    print(f"Loading images from: {folder_path}...")
    if not os.path.exists(folder_path):
        print(f"FATAL ERROR: Path does not exist -> {folder_path}")
        return None
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if i >= limit: break
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (128, 128)))
    print(f"-> Loaded {len(images)} images.")
    return np.array(images)

def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    return np.array([graycoprops(glcm, prop).mean() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']])

def extract_sift_features(image):
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image, None)
    return np.zeros(128) if descriptors is None else descriptors.mean(axis=0)

def create_tampered_images(images):
    tampered_images = []
    for img in images:
        method = np.random.choice(['noise', 'contrast', 'watermark'])
        tampered_img = img.copy()
        if method == 'noise':
            s_vs_p = 0.5; amount = 0.04; num_salt = np.ceil(amount * img.size * s_vs_p); coords = [np.random.randint(0, i-1, int(num_salt)) for i in img.shape]
            if coords[0].size > 0: tampered_img[tuple(coords)] = 255
            num_pepper = np.ceil(amount*img.size*(1.-s_vs_p)); coords = [np.random.randint(0, i-1, int(num_pepper)) for i in img.shape]
            if coords[0].size > 0: tampered_img[tuple(coords)] = 0
        elif method == 'contrast': tampered_img = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
        elif method == 'watermark': cv2.putText(tampered_img, 'Cisco', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128), 2, cv2.LINE_AA)
        tampered_images.append(tampered_img)
    return np.array(tampered_images)

def extract_firewall_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    mean, std_dev = cv2.meanStdDev(image)
    return np.hstack([hist, mean.flatten(), std_dev.flatten()])

# ================================================================
#               2. CONFIGURE YOUR FILE PATHS HERE
# ================================================================
# --- IMPORTANT: UPDATE THESE THREE FOLDER PATHS ---
folder_benign = r"C:\MEEE\BREAST CANCER PREDICTION\archive (3)\CLAHE_images\benign"
folder_malignant = r"C:\MEEE\BREAST CANCER PREDICTION\archive (3)\CLAHE_images\malignant"
folder_normal = r"C:\MEEE\BREAST CANCER PREDICTION\archive (3)\CLAHE_images\normal"

# ================================================================
#               3. LOAD DATA AND BUILD ARTIFACTS
# ================================================================
# --- Load data for scaler and pca ---
images_benign = load_images_from_folder(folder_benign)
images_malignant = load_images_from_folder(folder_malignant)
images_normal = load_images_from_folder(folder_normal)

if images_benign is None or images_malignant is None or images_normal is None:
    print("\nExiting due to path error. Please fix paths at the top of the script.")
else:
    all_images = np.concatenate([images_benign, images_malignant, images_normal], axis=0)
    print(f"\nTotal images loaded for processing: {len(all_images)}")

    # --- Build scaler and pca ---
    print("\nExtracting features for Scaler and PCA...")
    combined_features = np.hstack(([extract_glcm_features(img) for img in all_images], [extract_sift_features(img) for img in all_images]))
    
    print("Fitting and saving Scaler...")
    scaler = StandardScaler().fit(combined_features)
    joblib.dump(scaler, 'scaler.joblib')
    print("--> scaler.joblib SAVED.")

    print("Fitting and saving PCA...")
    X_scaled = scaler.transform(combined_features)
    pca = PCA(n_components=0.95).fit(X_scaled)
    joblib.dump(pca, 'pca.joblib')
    print("--> pca.joblib SAVED.")

    # --- Build AI Firewall ---
    print("\nBuilding AI Firewall...")
    firewall_tampered_images = create_tampered_images(images_normal)
    firewall_images = np.concatenate([images_normal, firewall_tampered_images])
    firewall_labels = np.concatenate([np.zeros(len(images_normal)), np.ones(len(firewall_tampered_images))])
    firewall_features = np.array([extract_firewall_features(img) for img in firewall_images])

    print("Training and saving AI Firewall model...")
    firewall_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42).fit(firewall_features, firewall_labels)
    joblib.dump(firewall_model, 'ai_firewall_model.joblib')
    print("--> ai_firewall_model.joblib SAVED.")

    print("\n--- MASTER BUILD SCRIPT FINISHED SUCCESSFULLY! ---")
    print("All three .joblib files have been created in your project folder.")
    # ================================================================
#               4. BUILD THE MAIN DEEP LEARNING MODEL
# ================================================================

print("\n--- Building Main Deep Learning Model (PyTorch) ---")

# Add necessary PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split as pytorch_train_test_split

# --- Define the Model Architecture ---
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

# --- Prepare Data for PyTorch ---
# Create the main labels array
labels_benign = np.full(len(images_benign), 0)
labels_malignant = np.full(len(images_malignant), 1)
labels_normal = np.full(len(images_normal), 2)
all_labels = np.concatenate([labels_benign, labels_malignant, labels_normal])

# We already have X_scaled and pca from the previous steps
X_pca = pca.transform(X_scaled)

X_train, X_test, y_train, y_test = pytorch_train_test_split(X_pca, all_labels, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# --- Initialize and Train the Model ---
input_dim = X_pca.shape[1]
output_dim = len(np.unique(all_labels))
model = NeuralNet14Layer(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting PyTorch model training...")
num_epochs = 20 # Using 20 epochs as in your original code
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished.")

# --- Save the Trained Model ---
torch.save(model.state_dict(), '14_layer_neural_network.pth')
print("--> 14_layer_neural_network.pth SAVED.")

print("\n--- MASTER BUILD SCRIPT FINISHED SUCCESSFULLY! ---")
print("All four model files have been created in your project folder.")