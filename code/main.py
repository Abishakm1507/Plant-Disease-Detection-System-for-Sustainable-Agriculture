import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load model and predict the disease
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 Home", "🔍 Disease Recognition"])

# Main Page Logic
if app_mode == "🏠 Home":
    st.markdown(
        "<h1 style='text-align: center; color: green;'>Plant Disease Detection System 🌿</h1><br>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<h3 style='text-align:center; color: darkgreen;'>🌱 Welcome to the Plant Disease Detection System! 🌱 </h3>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size: 18px; line-height: 1.6;'>Plant diseases threaten crop yield and food security worldwide. Our system leverages the power of deep learning to help farmers and plant enthusiasts quickly identify diseases and take timely actions to protect their crops and plants. <br>This application is designed to make plant disease detection easy, fast, and accessible. With just a few clicks, you can upload an image, get results, and start taking action!</p>", 
        unsafe_allow_html=True
    )

    # How It Works Section
    st.markdown("---")
    st.subheader("🌟 How It Works")
    st.markdown(
        """
        1. **Upload an Image**: On the 'Disease Recognition' page, upload a clear image of the plant leaf.
        2. **Analyze the Image**: The system uses a trained deep learning model to analyze the uploaded image.
        3. **Get Results**: The model predicts the type of disease (if any) affecting the plant and displays the result.
        4. **Take Action**: Use the provided information to take necessary steps to treat the plant.
        """,
        unsafe_allow_html=True
    )

    # Display the home page images in two columns
    col1, col2 = st.columns(2)  # Create two columns
    
    with col1:
        img_home = Image.open("image.png")
        st.image(img_home, caption="Healthy Plants", width=400)
        
    with col2:
        img_disease = Image.open("disease.png")
        st.image(img_disease, caption="Diseased Plants", width=400)

    # Learn More Button
    st.markdown("---")
    st.subheader("🌟 Learn More About Our Technology")
    st.markdown(
        """
        <p style='font-size: 18px; line-height: 1.6;'> 
        Our system utilizes cutting-edge deep learning algorithms to analyze images of plant leaves and detect potential diseases. 
        Here's how it works:
        </p>
        <ul style='font-size: 18px;'>
            <li><b>Deep Learning Models:</b> We use Convolutional Neural Networks (CNNs), a type of deep learning model known for its excellent performance in image recognition tasks. Our CNN is trained to recognize patterns in plant leaves and classify them into healthy or diseased categories.</li>
            <li><b>Training on Large Datasets:</b> The model is trained on a vast dataset of plant images that include different types of diseases. This dataset contains thousands of labeled images, enabling the model to learn and generalize across various plant species and disease types.</li>
            <li><b>Real-Time Prediction:</b> Once the model is trained, users can upload an image of a plant leaf, and the system will process the image in real-time to predict any disease. The prediction is fast and accurate, providing immediate results for timely action.</li>
            <li><b>Technology Stack:</b> Our application is built with TensorFlow, an open-source machine learning library, and Streamlit for building interactive web applications.</li>
        </ul>
        <p style='font-size: 18px; line-height: 1.6;'>The combination of deep learning and user-friendly interface allows both experts and non-experts to detect plant diseases quickly and efficiently, helping to prevent the spread of diseases and ensuring healthier crops.</p>
        """, unsafe_allow_html=True)

    img_tech = Image.open("Exp_image.png")  # Replace with the actual image path
    st.image(img_tech, caption="Our Deep Learning Model in Action", width=600)  # Adjust image width as needed


elif app_mode == "🔍 Disease Recognition":
    # Display the disease recognition page image
    img_disease = Image.open("Diseases.png")
    st.image(img_disease)

    st.header("🔍 Disease Recognition")
    st.markdown("Upload an image of the plant leaf to detect possible diseases.")

    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, width=400, use_column_width=True)  # Adjust width and alignment
    
    # Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy']
        st.success(f"Model is predicting it's a {class_name[result_index]}")
