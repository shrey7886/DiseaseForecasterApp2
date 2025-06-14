import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelweights")
model_path = os.path.join(model_dir, "newmodel_30.pth")

class_labels = {
    0: "Meningioma Tumor",
    1: "Normal (No Tumor)",
    2: "Glioma Tumor",
    3: "Pituitary Tumor",
    4: "Other"
}

# Add confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.7

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(6 * 6 * 128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache_data
def load_model(model_path, num_classes):
    model = Model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def validate_image(image):
    """Validate if the uploaded image is suitable for analysis."""
    try:
        # Check if image is too small
        if image.size[0] < 100 or image.size[1] < 100:
            return False, "Image is too small. Please upload a larger image (minimum 100x100 pixels)."
        
        # Check if image is too large (increased limit to 4000x4000)
        if image.size[0] > 4000 or image.size[1] > 4000:
            # Instead of rejecting, we'll resize large images
            max_size = 4000
            ratio = min(max_size / image.size[0], max_size / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            return True, "Image was automatically resized to fit within size limits."
        
        # Check if image is grayscale
        if image.mode == 'L':
            return False, "Grayscale images are not supported. Please upload a color image."
        
        return True, "Image is valid."
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def preprocess_image(image):
    """Preprocess the image for model input with error handling."""
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio
        target_size = (224, 224)
        image.thumbnail((max(target_size), max(target_size)), Image.Resampling.LANCZOS)
        
        # Create a new image with padding
        new_image = Image.new('RGB', target_size, (0, 0, 0))
        new_image.paste(image, ((target_size[0] - image.size[0]) // 2,
                              (target_size[1] - image.size[1]) // 2))
        
        # Use the same normalization as during training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Add logging for debugging
        st.write("Image size after preprocessing:", new_image.size)
        st.write("Image mode:", new_image.mode)
        
        return transform(new_image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def get_prediction_with_confidence(model, input_tensor):
    """Get model prediction with confidence scores and detailed analysis."""
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
        
        # Get top 3 predictions
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        # Detailed logging for debugging
        st.write("Raw model outputs (logits):", output[0].tolist())
        st.write("Class probabilities:", probabilities.tolist())
        
        # Show top 3 predictions
        st.write("Top 3 predictions:")
        for prob, idx in zip(top3_prob, top3_indices):
            st.write(f"- {class_labels[idx.item()]}: {prob.item()*100:.2f}%")
        
        # Special handling for pituitary tumors
        pituitary_prob = probabilities[3].item()  # Class 3 is pituitary
        if pituitary_prob > 0.3 and prediction != 3:
            st.warning("⚠️ Note: The image shows some characteristics of a pituitary tumor, but the model is not confident enough to make this the primary prediction.")
        
        return prediction, confidence, probabilities

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.title("Brain Tumor Classification System")
st.write("Upload a brain MRI image to classify the type of tumor (if any).")

# Sidebar with upload
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if os.path.isfile(model_path):
    model = load_model(model_path, num_classes=5)
    
    if uploaded_file is not None:
        try:
            # Load and validate image
            image = Image.open(uploaded_file)
            is_valid, message = validate_image(image)
            
            if not is_valid:
                st.error(message)
            else:
                # Display original image
                st.image(image, caption='Uploaded MRI', use_container_width=True)
                
                # Preprocess and predict
                input_tensor = preprocess_image(image)
                if input_tensor is not None:
                    prediction, confidence, probabilities = get_prediction_with_confidence(model, input_tensor)
                    predicted_class = class_labels.get(prediction, "Unknown")
                    
                    # Display results in a simpler format
                    st.subheader("Classification Results")
                    st.write(f'**Predicted Class:** {predicted_class}')
                    st.write(f'**Confidence:** {confidence*100:.2f}%')
                    
                    # Show warning if confidence is low
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning(f"⚠️ Low confidence prediction ({confidence*100:.2f}%). Please verify the result.")
                    
                    # Display probability bars for all classes
                    st.subheader("Class Probabilities")
                    for i, prob in enumerate(probabilities):
                        if prob > 0.1:  # Only show probabilities above 10%
                            st.write(f"{class_labels[i]}: {prob.item()*100:.2f}%")
                            st.progress(float(prob.item()))  # Convert tensor to float
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
else:
    st.error("Model file not found. Please ensure the model weights file exists at the specified path.")
