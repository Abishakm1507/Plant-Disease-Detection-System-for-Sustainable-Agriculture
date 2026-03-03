# 🌱 Plant Disease Detection System

## 📌 Project Overview

The **Plant Disease Detection System** is an AI-powered web application designed to support sustainable agriculture through early and accurate plant disease identification.

By leveraging Deep Learning (Convolutional Neural Networks), the system detects diseases from plant leaf images, helping farmers reduce crop losses and improve yield.

Early disease detection plays a crucial role in modern precision agriculture, ensuring healthier crops and better productivity.

---

## 🎯 Key Objectives

- ✅ Develop a Convolutional Neural Network (CNN) for plant disease classification  
- ✅ Build a user-friendly web application for real-time disease prediction  
- ✅ Promote sustainable agricultural practices through AI-driven diagnosis  

---

## 🔬 Methodology

### 📂 1. Dataset Preparation
- Images resized to **128x128 pixels**
- Data preprocessing and normalization applied
- Images batched for efficient training
- Multi-class labeled dataset used for classification

### 🧠 2. Model Development
- Sequential CNN architecture
- Convolutional layers for feature extraction
- ReLU activation functions
- MaxPooling layers for dimensionality reduction
- Fully connected dense layers for classification

### 🏋️ 3. Training Process
- Loss Function: `Categorical Cross-Entropy`
- Optimizer: `Adam`
- Performance Metric: `Accuracy`
- Achieved model accuracy of **97%**

### 🌐 4. Web Application Deployment
- Built using **Streamlit**
- Users can upload plant leaf images
- Model predicts disease class instantly
- Clean and interactive UI

---

## 📊 Results

- 📈 Model Accuracy: **97%**
- 🚀 Fast real-time predictions
- 🌾 Practical usability for farmers and agriculture professionals
- 💡 Supports early disease detection to prevent crop loss

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Streamlit
- Matplotlib

---

## 🚀 How to Run the Project

### 1️. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-detection.git
cd plant-disease-detection
cd code
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🌟 Conclusion

The Plant Disease Detection System demonstrates how Artificial Intelligence can significantly enhance agricultural practices. By enabling accurate and timely plant disease diagnosis, the system helps farmers reduce losses, improve productivity, and move towards sustainable farming solutions.

This project highlights the impact of AI in real-world problem solving and showcases the integration of Deep Learning with practical web deployment.
