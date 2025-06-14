import streamlit as st
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Disease Forecaster App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 10px 0;
    }
    .prediction-text {
        color: #1E3A8A;
        font-size: 1.2em;
        font-weight: bold;
    }
    .probability-text {
        color: #2563EB;
        font-size: 1.1em;
    }
    .warning-box {
        background-color: #FEF2F2;
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #DC2626;
        margin: 10px 0;
    }
    .warning-text {
        color: #991B1B;
        font-size: 1.1em;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    model_dir = os.path.join(os.path.dirname(__file__), 'modelweights')
    heart_model = joblib.load(os.path.join(model_dir, 'gaussian_nb_model.pkl'))
    heart_scaler = joblib.load(os.path.join(model_dir, 'gaussian_nb_scaler.pkl'))
    return heart_model, heart_scaler

# Constants
FEATURE_ORDER = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 
                 'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH', 
                 'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N', 
                 'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']

CLASS_LABELS = {
    0: "Meningioma Tumor",
    1: "Normal (No Tumor)",
    2: "Glioma Tumor",
    3: "Pituitary Tumor",
    4: "Other"
}

CONFIDENCE_THRESHOLD = 0.7

# Model class definition
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
def load_brain_model(model_path, num_classes):
    model = Model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        target_size = (224, 224)
        image.thumbnail((max(target_size), max(target_size)), Image.Resampling.LANCZOS)
        
        new_image = Image.new('RGB', target_size, (0, 0, 0))
        new_image.paste(image, ((target_size[0] - image.size[0]) // 2,
                              (target_size[1] - image.size[1]) // 2))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(new_image).unsqueeze(0)
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def get_prediction_with_confidence(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item()
        
        top3_prob, top3_indices = torch.topk(probabilities, 3)
        
        return prediction, confidence, probabilities, top3_prob, top3_indices

def main():
    st.title('🏥 Disease Forecaster App')
    st.markdown("""
    Welcome to the Disease Forecaster App! This application helps you predict:
    - Heart Disease Risk
    - Brain Tumor Classification from MRI Images
    
    Select a mode from the sidebar to get started.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Heart Disease Prediction", "Brain Tumor Detection"],
        help="Choose between heart disease risk prediction or brain tumor classification"
    )

    if app_mode == "Heart Disease Prediction":
        st.header('❤️ Heart Disease Risk Prediction')
        st.markdown("""
        This tool helps predict the risk of heart disease based on various health parameters.
        Please fill in your details below.
        """)

        # Create two columns for input fields
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Personal Information")
            age = st.slider('Age', 10, 80, 40, 
                          help="Enter your age in years")
            sex = st.radio('Sex', ['Male', 'Female'],
                         help="Select your biological sex")
            chest_pain_type = st.selectbox(
                'Chest Pain Type',
                ['Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic', 'Typical Angina'],
                help="Select the type of chest pain you experience"
            )

        with col2:
            st.subheader("Medical Measurements")
            resting_bp = st.slider('Resting Blood Pressure (mm Hg)', 90, 200, 140,
                                 help="Your resting blood pressure in mm Hg")
            cholesterol = st.slider('Cholesterol (mg/dl)', 100, 400, 289,
                                  help="Your cholesterol level in mg/dl")
            fasting_bs = st.selectbox('Fasting Blood Sugar',
                                    ['< 120 mg/dl', '> 120 mg/dl'],
                                    help="Your fasting blood sugar level")

        # Second row of inputs
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Additional Measurements")
            resting_ecg = st.selectbox(
                'Resting ECG Results',
                ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'],
                help="Results from your resting ECG test"
            )
            max_hr = st.slider('Maximum Heart Rate', 70, 210, 172,
                             help="Your maximum heart rate achieved during exercise")
            exercise_angina = st.selectbox(
                'Exercise Induced Angina',
                ['No', 'Yes'],
                help="Whether you experience angina during exercise"
            )

        with col4:
            st.subheader("Additional Parameters")
            oldpeak = st.slider('ST Depression (Oldpeak)', 0.0, 6.0, 0.0,
                              help="ST depression induced by exercise relative to rest")
            st_slope = st.selectbox(
                'ST Slope',
                ['Upsloping', 'Flat', 'Downsloping'],
                help="The slope of the peak exercise ST segment"
            )

        # Convert inputs to model format
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

        # Make prediction
        if st.button('Predict Heart Disease Risk', type='primary'):
            try:
                heart_model, heart_scaler = load_models()
                input_features = [user_input[feature] for feature in FEATURE_ORDER]
                input_features_array = np.array(input_features).reshape(1, -1)
                input_features_scaled = heart_scaler.transform(input_features_array)
                prediction = heart_model.predict(input_features_scaled)[0]
                prediction_proba = heart_model.predict_proba(input_features_scaled)[0]

                # Display results
                st.markdown("### Prediction Results")
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f'<p class="prediction-text">Prediction: {"High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"}</p>', unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown('<p class="warning-text">⚠️ Based on the provided information, there is a high risk of heart disease. Please consult with a healthcare professional for proper evaluation.</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown('<p class="warning-text">✅ Based on the provided information, there is a low risk of heart disease. However, please maintain regular check-ups with your healthcare provider.</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                with result_col2:
                    st.markdown("### Risk Probability")
                    risk_prob = prediction_proba[1] * 100
                    st.progress(float(risk_prob/100))
                    st.write(f"Probability of Heart Disease: {risk_prob:.1f}%")

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

    elif app_mode == "Brain Tumor Detection":
        st.header('🧠 Brain Tumor Classification')
        st.markdown("""
        This tool helps classify brain tumors from MRI images. Upload a clear MRI image to get started.
        """)

        # Sidebar for image upload
        st.sidebar.title("Upload MRI Image")
        st.sidebar.markdown("""
        ### Instructions
        1. Upload a clear MRI image (JPG, JPEG, or PNG)
        2. Image should be of a brain MRI scan
        3. For best results, use a well-lit, clear image
        """)

        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a brain MRI image for classification"
        )

        model_path = os.path.join(os.path.dirname(__file__), "modelweights", "newmodel_30.pth")

        if os.path.isfile(model_path):
            model = load_brain_model(model_path, num_classes=5)
            
            if uploaded_file is not None:
                try:
                    # Load and display image
                    image = Image.open(uploaded_file)
                    st.image(image, caption='Uploaded MRI', use_container_width=True)
                    
                    # Preprocess and predict
                    input_tensor = preprocess_image(image)
                    if input_tensor is not None:
                        with st.spinner('Analyzing image...'):
                            prediction, confidence, probabilities, top3_prob, top3_indices = get_prediction_with_confidence(model, input_tensor)
                            predicted_class = CLASS_LABELS.get(prediction, "Unknown")
                            
                            # Display results
                            st.markdown("### Classification Results")
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                                st.markdown(f'<p class="prediction-text">Predicted Class: {predicted_class}</p>', unsafe_allow_html=True)
                                
                                if confidence < CONFIDENCE_THRESHOLD:
                                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                                    st.markdown('<p class="warning-text">⚠️ Low confidence prediction. Please consult a medical professional.</p>', unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)

                            with result_col2:
                                st.markdown("### Top 3 Predictions")
                                for prob, idx in zip(top3_prob, top3_indices):
                                    st.write(f"- {CLASS_LABELS[idx.item()]}: {prob.item()*100:.1f}%")
                                    st.progress(float(prob.item()))

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        else:
            st.error("Model file not found. Please ensure the model weights file exists.")

if __name__ == '__main__':
    main()
