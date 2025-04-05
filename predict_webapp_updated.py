import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

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
    # Load and display image for annotation
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    # Create canvas for manual selection
    st.markdown("**Draw a rectangle around the wastewater area:**")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="rect",
        key="canvas",
    )

    # Process rectangle coordinates
    if canvas_result.json_data is not None:
        rectangles = canvas_result.json_data.get("objects", [])
        if len(rectangles) > 0:
            # Get latest rectangle
            rect = rectangles[-1]
            x = int(rect["left"])
            y = int(rect["top"])
            w = int(rect["width"])
            h = int(rect["height"])
            
            # Validate selection
            if w > 0 and h > 0:
                # Extract ROI
                roi = img_np[y:y+h, x:x+w]
                
                # Convert to HSV and calculate average
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                avg_hsv = np.mean(hsv_roi, axis=(0, 1))
                hue, saturation, value = avg_hsv
                
                # Display selection results
                annotated_img = img_np.copy()
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                st.image(annotated_img, caption="Selected Wastewater Area", use_column_width=True)
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
                st.error("Invalid selection! Please draw a proper rectangle.")
        else:
            st.warning("No selection detected! Draw a rectangle around the wastewater area.")
    else:
        st.warning("Please draw a rectangle around the wastewater area.")
else:
    st.warning("Please upload an image to proceed.")
