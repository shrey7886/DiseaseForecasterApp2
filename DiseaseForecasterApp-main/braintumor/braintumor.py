import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


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

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


st.sidebar.title("Brain Tumor Classification")
st.sidebar.write("Upload an MRI image and get the classification result.")

model_path = "../modelweights/newmodel_30.pth"

class_labels = {
    0: "Meningioma Tumor",
    1: "Normal (No Tumor)",
    2: "Glioma Tumor",
    3: "Pituitary Tumor"
}

if os.path.isfile(model_path):
    model = load_model(model_path, num_classes=5)
    
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=False)
        
        input_tensor = preprocess_image(image)
        
        with st.spinner('Making prediction...'):
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()
                predicted_class = class_labels.get(prediction, "Unknown")
                confidence = torch.softmax(output, dim=1)[0] *100
        
        st.write(f'Predicted class: **{predicted_class}**')
        st.write(f'Confidence: **{confidence[prediction]:.8f}%**')
        
        st.write("**Class Probabilities:**")
        for i, label in class_labels.items():
            st.write(f'{label}: {confidence[i]:.8f}%')
else:
    st.sidebar.write("Please enter a valid model weights file path.")
