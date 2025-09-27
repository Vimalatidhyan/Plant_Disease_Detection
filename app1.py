# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# -----------------------
# 1️⃣ Load Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 36  # Number of classes
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("crop_model_resnet50.pth", map_location=device))
model = model.to(device)
model.eval()

# -----------------------
# 2️⃣ Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------
# 3️⃣ Classes & Disease Info
# -----------------------
classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "EggPlant_Healthy Leaf", "EggPlant_Insect Pest Disease", "EggPlant_Leaf Spot Disease",
    "EggPlant_Mosaic Virus Disease", "EggPlant_Small Leaf Disease", "EggPlant_White Mold Disease",
    "EggPlant_Wilt Disease", "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Rice_Bacterial Leaf Blight", "Rice_Brown Spot", "Rice_Healthy Rice Leaf",
    "Rice_Leaf Blast", "Rice_Leaf scald", "Rice_Sheath Blight",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

disease_info = {
    "Tomato__Tomato_YellowLeaf_Curl_Virus": {
        "Description": "Leaves curl, yellow, and plant growth is stunted.",
        "Cause": "Spread by whiteflies carrying the virus.",
        "Prevention": "Use resistant varieties, control whiteflies, remove infected plants."
    },
    "Tomato_Septoria_leaf_spot": {
        "Description": "Small dark spots appear on older leaves.",
        "Cause": "Fungal infection (Septoria lycopersici) thrives in humid conditions.",
        "Prevention": "Apply fungicides, avoid overhead watering, rotate crops."
    },
    "Tomato_Early_blight": {
        "Description": "Concentric brown spots on leaves and fruit.",
        "Cause": "Fungus Alternaria solani.",
        "Prevention": "Use resistant seeds, crop rotation, fungicides."
    },
    "Rice_Leaf Blast": {
        "Description": "Spindle-shaped lesions on leaves, affects yield.",
        "Cause": "Fungus Magnaporthe oryzae.",
        "Prevention": "Plant resistant varieties, balanced fertilizer, fungicides."
    },
    "Potato___Late_blight": {
        "Description": "Water-soaked leaf lesions, can destroy crops.",
        "Cause": "Phytophthora infestans fungus.",
        "Prevention": "Resistant varieties, fungicides, destroy infected plants."
    },
    "pepper,_bell__Bacterial_spot": {
        "Description": "Dark, water-soaked spots on leaves and fruit.",
        "Cause": "Bacteria Xanthomonas spread by rain and irrigation.",
        "Prevention": "Clean seeds, copper sprays, avoid working in wet fields."
    },
    "EggPlant_Mosaic Virus Disease": {
        "Description": "Mottled, mosaic patterns on leaves.",
        "Cause": "Spread by insects and mechanical contact.",
        "Prevention": "Control insect vectors, virus-free planting material."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Description": "Cigar-shaped lesions on maize leaves.",
        "Cause": "Fungus Exserohilum turcicum.",
        "Prevention": "Resistant hybrids, crop rotation, fungicides."
    },
    "Apple___healthy": {
        "Description": "Leaf shows no disease symptoms.",
        "Cause": "Healthy apple plant.",
        "Prevention": "Maintain good orchard practices."
    },
    "Tomato__Tomato_mosaic_virus": {
        "Description": "Mottled, discolored leaves, reduced yield.",
        "Cause": "Tobamovirus spread via seeds and handling.",
        "Prevention": "Certified seeds, disinfect tools."
    },
    "Tomato_Leaf_Mold": {
        "Description": "Yellow spots on upper leaf surface, mold underneath.",
        "Cause": "Fungus Passalora fulva in humid conditions.",
        "Prevention": "Ventilate greenhouses, reduce humidity, fungicides."
    },
    "Tomato_Bacterial_spot": {
        "Description": "Greasy dark spots on leaves, stems, fruit.",
        "Cause": "Bacteria Xanthomonas campestris.",
        "Prevention": "Copper sprays, crop rotation, pathogen-free seeds."
    },
    "Rice_Healthy Rice Leaf": {
        "Description": "No visible disease symptoms.",
        "Cause": "Healthy rice plant.",
        "Prevention": "Good field management."
    },
    "Potato___healthy": {
        "Description": "Leaves show no disease.",
        "Cause": "Healthy potato plant.",
        "Prevention": "Disease-free seeds, proper field care."
    },
    "EggPlant_Wilt Disease": {
        "Description": "Leaves and stems suddenly wilt.",
        "Cause": "Soil-borne fungi or bacteria attacking roots.",
        "Prevention": "Resistant varieties, crop rotation."
    },
    "EggPlant_Leaf Spot Disease": {
        "Description": "Circular brown to black spots on leaves.",
        "Cause": "Fungal pathogens like Cercospora.",
        "Prevention": "Fungicides, crop rotation, clean seeds."
    },
    "Corn_(maize)___healthy": {
        "Description": "No visible disease symptoms.",
        "Cause": "Healthy maize plant.",
        "Prevention": "Good field practices."
    },
    "Apple___Cedar_apple_rust": {
        "Description": "Bright orange spots on leaves.",
        "Cause": "Fungus Gymnosporangium juniperi-virginianae.",
        "Prevention": "Remove nearby junipers, fungicides."
    },
    "Tomato__Target_Spot": {
        "Description": "Spots with concentric rings on leaves.",
        "Cause": "Fungus Corynespora cassiicola.",
        "Prevention": "Fungicides, remove infected plant debris."
    },
    "Tomato_Late_blight": {
        "Description": "Dark lesions on leaves, stems, and fruit.",
        "Cause": "Phytophthora infestans fungus.",
        "Prevention": "Resistant varieties, fungicides, remove infected plants."
    },
    "Rice_Sheath Blight": {
        "Description": "Gray-green lesions at the leaf base.",
        "Cause": "Fungus Rhizoctonia solani.",
        "Prevention": "Proper spacing, fungicides, crop rotation."
    },
    "Rice_Brown Spot": {
        "Description": "Brown lesions on leaves and grains.",
        "Cause": "Fungus Cochliobolus miyabeanus.",
        "Prevention": "Balanced fertilization, resistant varieties, fungicides."
    },
    "Potato___Early_blight": {
        "Description": "Dark concentric spots on leaves, stems, tubers.",
        "Cause": "Alternaria solani fungus.",
        "Prevention": "Resistant varieties, crop rotation, fungicides."
    },
    "EggPlant_White Mold Disease": {
        "Description": "White cottony mold on leaves and stems.",
        "Cause": "Sclerotinia sclerotiorum fungus.",
        "Prevention": "Remove infected plants, fungicides."
    },
    "EggPlant_Insect Pest Disease": {
        "Description": "Leaves damaged by insects.",
        "Cause": "Various insect pests.",
        "Prevention": "Use insecticides and monitor regularly."
    },
    "Corn_(maize)___Common_rust_": {
        "Description": "Reddish-brown pustules on leaves.",
        "Cause": "Fungus Puccinia sorghi.",
        "Prevention": "Resistant varieties, fungicides."
    },
    "Apple___Black_rot": {
        "Description": "Dark lesions on leaves and fruit.",
        "Cause": "Fungus Guignardia bidwellii.",
        "Prevention": "Prune infected parts, fungicides."
    },
    "Tomato__Spider_mites_Two_spotted_spider_mite": {
        "Description": "Yellow stippling and webbing on leaves.",
        "Cause": "Spider mite infestation.",
        "Prevention": "Miticides, biological control, maintain humidity."
    },
    "Tomato__healthy": {
        "Description": "No disease symptoms.",
        "Cause": "Healthy tomato plant.",
        "Prevention": "Good field practices."
    },
    "Rice_Leaf scald": {
        "Description": "Water-soaked lesions that turn tan and spread.",
        "Cause": "Fungus Microdochium oryzae.",
        "Prevention": "Resistant varieties, balanced fertilization, fungicides."
    },
    "Rice_Bacterial Leaf Blight": {
        "Description": "Yellowing and wilting along leaf margins.",
        "Cause": "Bacteria Xanthomonas oryzae.",
        "Prevention": "Resistant varieties, clean water, avoid high nitrogen."
    },
    "Pepper,_bell___healthy": {
        "Description": "No disease symptoms.",
        "Cause": "Healthy pepper plant.",
        "Prevention": "Good cultivation practices."
    },
    "EggPlant_Small Leaf Disease": {
        "Description": "Leaves remain smaller than normal.",
        "Cause": "Nutrient deficiency or mild viral infection.",
        "Prevention": "Balanced fertilization, healthy seedlings."
    },
    "EggPlant_Healthy Leaf": {
        "Description": "No disease present.",
        "Cause": "Healthy eggplant plant.",
        "Prevention": "Good field management."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_lea...": {
        "Description": "Gray circular spots with dark borders on maize leaves.",
        "Cause": "Fungus Cercospora.",
        "Prevention": "Crop rotation, fungicides, remove infected debris."
    },
    "Apple___Apple_scab": {
        "Description": "Dark, scabby lesions on leaves and fruit.",
        "Cause": "Fungus Venturia inaequalis.",
        "Prevention": "Fungicides, remove infected leaves, resistant varieties."
    }
}


# -----------------------
# 4️⃣ Streamlit App
# -----------------------
st.title("Crop Disease Detection App")
st.write("Upload a leaf, stem, or fruit image to detect the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess & Predict
    input_image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs, 1)

    disease = classes[predicted.item()]
    info = disease_info.get(disease, "No information available")

st.success(f"Predicted Disease: **{disease}**")
st.info(f"Description & Prevention: {info}")


