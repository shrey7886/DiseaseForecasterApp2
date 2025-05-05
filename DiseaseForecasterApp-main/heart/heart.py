import streamlit as st
import joblib
import numpy as np
import pandas as pd

loaded_model = joblib.load('"C:\ML Projects\DiseaseForecasterApp-main1\DiseaseForecasterApp-main\modelweights\gaussian_nb_model.pkl"')
scaler = joblib.load('"C:\ML Projects\DiseaseForecasterApp-main1\DiseaseForecasterApp-main\modelweights\gaussian_nb_scaler.pkl"')  

feature_order = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP','ChestPainType_TA',
                 'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
                 'ExerciseAngina_N', 'ExerciseAngina_Y',
                 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']



def main():
    st.title('Heart Disease Prediction')

    age = st.slider('Age', 10, 80, 40)
    sex = st.selectbox('Sex', ['M', 'F'])
    chest_pain_type = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY','TA'])
    resting_bp = st.slider('Resting BP', 90, 200, 140)
    cholesterol = st.slider('Cholesterol', 100, 400, 289)
    fasting_bs = st.selectbox('Fasting Blood Sugar (0: <120 mg/dl, 1: >120 mg/dl)', [0, 1])
    resting_ecg = st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
    max_hr = st.slider('Max Heart Rate', 70, 210, 172)
    exercise_angina = st.selectbox('Exercise Induced Angina (N: No, Y: Yes)', ['N', 'Y'])
    oldpeak = st.slider('Oldpeak', 0.0, 6.0, 0.0)
    st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

    
    user_input = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'MaxHR': max_hr,
    'Oldpeak': oldpeak,
    'Sex_F': 1 if sex == 'F' else 0,  
    'Sex_M': 1 if sex == 'M' else 0,
    'ChestPainType_ASY': 1 if chest_pain_type == 'ASY' else 0,
    'ChestPainType_ATA': 1 if chest_pain_type == 'ATA' else 0,
    'ChestPainType_NAP': 1 if chest_pain_type == 'NAP' else 0,
    'ChestPainType_TA': 1 if chest_pain_type == 'TA' else 0,
    'RestingECG_LVH': 1 if resting_ecg == 'LVH' else 0,
    'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
    'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
    'ExerciseAngina_N': 1 if exercise_angina == 'N' else 0,
    'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
    'ST_Slope_Down': 1 if st_slope == 'Down' else 0,
    'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
    'ST_Slope_Up': 1 if st_slope == 'Up' else 0
}

    

    input_features = [user_input[feature] for feature in feature_order]
    input_features_array = np.array(input_features).reshape(1, -1)
    input_features_scaled = scaler.transform(input_features_array)


    prediction = loaded_model.predict(input_features_scaled)[0]

    st.write(f'Predicted Heart Disease Status: {prediction}')

if __name__ == '__main__':
    main()
