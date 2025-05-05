import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


loaded_model = joblib.load('modelweights/gaussian_nb_model.pkl')
scaler = joblib.load('modelweights/gaussian_nb_scaler.pkl')

feature_order = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP','ChestPainType_TA',
                 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
                 'ExerciseAngina_N', 'ExerciseAngina_Y',
                 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']

class_labels = {
    0: "Meningioma Tumor",
    1: "Normal (No Tumor)",
    2: "Glioma Tumor",
    3: "Pituitary Tumor"
}

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

def main():
    st.title('Disease Forecaster App')

    
    app_mode = st.sidebar.selectbox("Select Mode", ["Heart Disease Prediction", "Brain Tumor Detection"])

    if app_mode == "Heart Disease Prediction":
        st.header('Heart Disease Prediction')

        age = st.slider('Age', 10, 80, 40)
        sex = st.radio('Sex', ['Male', 'Female'])
        chest_pain_type = st.selectbox('Chest Pain Type', ['Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic','Typical Angina'])
        resting_bp = st.slider('Resting BP', 90, 200, 140)
        cholesterol = st.slider('Cholesterol', 100, 400, 289)
        fasting_bs = st.selectbox('Fasting Blood Sugar', ['< 120 mg/dl', '> 120 mg/dl'])
        resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
        max_hr = st.slider('Max Heart Rate', 70, 210, 172)
        exercise_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        oldpeak = st.slider('Oldpeak', 0.0, 6.0, 0.0)
        st_slope = st.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])

        user_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': 1 if fasting_bs == '> 120 mg/dl' else 0,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_F': 1 if sex == 'Female' else 0,  
            'Sex_M': 1 if sex == 'Male' else 0,
            'ChestPainType_ASY': 1 if chest_pain_type == 'Asymptomatic' else 0,
            'ChestPainType_ATA': 1 if chest_pain_type == 'Typical Angina' else 0,
            'ChestPainType_NAP': 1 if chest_pain_type == 'Non-Anginal Pain' else 0,
            'ChestPainType_TA': 1 if chest_pain_type == 'Atypical Angina' else 0,
            'RestingECG_LVH': 1 if resting_ecg == 'Left ventricular hypertrophy' else 0,
            'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
            'RestingECG_ST': 1 if resting_ecg == 'ST-T wave abnormality' else 0,
            'ExerciseAngina_N': 1 if exercise_angina == 'No' else 0,
            'ExerciseAngina_Y': 1 if exercise_angina == 'Yes' else 0,
            'ST_Slope_Down': 1 if st_slope == 'Downsloping' else 0,
            'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
            'ST_Slope_Up': 1 if st_slope == 'Upsloping' else 0
        }

        input_features = [user_input[feature] for feature in feature_order]
        input_features_array = np.array(input_features).reshape(1, -1)
        input_features_scaled = scaler.transform(input_features_array)

        prediction = loaded_model.predict(input_features_scaled)[0]

        st.subheader('Prediction Result')
        st.write(f'Predicted Heart Disease Status: {prediction}')

    elif app_mode == "Brain Tumor Detection":
        st.header('Brain Tumor Classification')

        
        st.sidebar.title("Upload MRI Image")
        st.sidebar.write("Upload an MRI image and get the classification result.")

        model_path = "modelweights/newmodel_30.pth"  

        if os.path.isfile(model_path):
            model = load_model(model_path, num_classes=5)
            
            uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image')
                
                input_tensor = preprocess_image(image)
                
                with st.spinner('Making prediction...'):
                    with torch.no_grad():
                        output = model(input_tensor)
                        prediction = torch.argmax(output, dim=1).item()
                        predicted_class = class_labels.get(prediction, "Unknown")
                        confidence = torch.softmax(output, dim=1)[0] * 100
                
                st.subheader('Prediction Result')
                st.write(f'Predicted Class: **{predicted_class}**')
                st.write(f'Confidence: **{confidence[prediction]:.2f}%**')

                st.subheader('Class Probabilities')
                for i, label in class_labels.items():
                    st.write(f'{label}: {confidence[i]:.2f}%')

        else:
            st.sidebar.write("Please enter a valid model weights file path.")

if __name__ == '__main__':
    main()
