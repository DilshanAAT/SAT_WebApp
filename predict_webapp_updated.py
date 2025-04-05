import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import cv2

# Load Model & Feature Columns
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit Page Config
st.set_page_config(page_title="Chemical Dosage Prediction", page_icon="🧪", layout="wide")

# Title
st.markdown('<h1 style="text-align: center; color: green;">Chemical Dosage Prediction</h1>', unsafe_allow_html=True)

# Upload Image Section
st.header("Upload Wastewater Image", anchor="upload_image")
uploaded_file = st.file_uploader("Upload Wastewater Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # Display the image
    st.image(image, caption="Original Image", use_column_width=True)
    
    # Manual ROI selection using sliders
    st.subheader("Select Region of Interest")
    
    # Get image dimensions
    height, width = img_np.shape[:2]
    
    # Create sliders for ROI selection
    col1, col2 = st.columns(2)
    with col1:
        x_start = st.slider("X Start", 0, width-10, int(width*0.25))
        y_start = st.slider("Y Start", 0, height-10, int(height*0.25))
    
    with col2:
        x_end = st.slider("X End", x_start+10, width, int(width*0.75))
        y_end = st.slider("Y End", y_start+10, height, int(height*0.75))
    
    # Extract and display ROI
    roi = img_np[y_start:y_end, x_start:x_end]
    
    # Convert to HSV and calculate average
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    avg_hsv = np.mean(hsv_roi, axis=(0, 1))
    hue, saturation, value = avg_hsv
    
    # Display selected region
    annotated_img = img_np.copy()
    cv2.rectangle(annotated_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
    st.image(annotated_img, caption="Selected Wastewater Area", use_column_width=True)
    
    # Show ROI closeup
    st.image(roi, caption="Closeup of Selected Area", use_column_width=True)
    
    st.success(f"Detected Color - Hue: {hue:.2f}, Saturation: {saturation:.2f}, Value: {value:.2f}")
    
    # Prediction inputs
    flow_rate = st.number_input("Flow Rate (m³/h):", min_value=0.0, step=0.1)
    
    if st.button("Predict Chemical Dosage"):
        input_data = pd.DataFrame([[hue, saturation, value]], columns=feature_columns)
        prediction = model.predict(input_data)[0]
        predict_val = prediction / 4
        
        # Calculations
        percentage_ratio = (predict_val / 500) * 100
        required_chemical_liters = (predict_val / 500) * flow_rate * 1000
        required_chemical_cubic_meters = required_chemical_liters / 1000
        
        # Display results
        st.success(f"Chemical Required: {predict_val:.2f} ml (per 500ml wastewater)")
        st.info(f"Dosage: {percentage_ratio:.2f}% of wastewater volume")
        st.warning(f"Estimated for {flow_rate} m³/h: {required_chemical_liters:.2f} L/h | {required_chemical_cubic_meters:.5f} m³/h")
else:
    st.warning("Please upload an image to proceed.")
