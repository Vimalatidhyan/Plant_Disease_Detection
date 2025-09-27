# Plant_Disease_Detection
AI-Based Crop Disease Detection using ResNet-50.
# Crop Disease Detection App

## 📝 Problem Statement
Crop diseases significantly affect farmers’ productivity and livelihood. Early detection of plant diseases can help prevent crop loss and improve yield.

## 🎯 Objective
To build an AI-based system that can identify diseases in crops from images of leaves, stems, or fruits and provide information about causes, prevention, and cure.

## 📂 Dataset
The dataset contains images of 10 crops with 36 different classes (healthy + multiple diseases per crop), including:
- Apple, Corn, Eggplant, Pepper, Potato, Rice, Tomato, etc.
- Each class has several images (original + augmented) to improve model accuracy.
- The dataset is split into *train* and *validation* folders.

## 🛠 Technology and Softwares Used
- *Programming Language:* Python  
- *Libraries:* PyTorch, Torchvision, NumPy, Pandas, Pillow, Streamlit  
- *Model Architecture:* ResNet-50 (CNN)  
- *Platform for Training:* Google Colab (GPU – NVIDIA T4)  
- *Version Control & Storage:* Google Drive  
- *Web Application Framework:* Streamlit  
- *Deployment:* Laptop / Web browser  

## ⚙ How It Works
1. Users upload an image of a crop leaf, stem, or fruit.  
2. The image is preprocessed and resized to 224x224 pixels.  
3. ResNet-50 model predicts the disease class.  
4. The app displays:
   - Predicted disease name  
   - Cause of the disease  
   - Preventive measures  
   - Recommended treatment  

## 📦 Installation
1. Clone this repository or download the files.
2. Install dependencies:
   ```bash
   pip install torch torchvision pillow streamlit
## To Run
Streamlit run app1.py

   

