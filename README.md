Sleep Disorder Prediction App
Predict sleep disorders like Insomnia, Sleep Apnea, or Parasomnia using health and lifestyle features. Built with SVM Classifier (best model) and deployed as a Streamlit web app.
Streamlit App
Predict your risk of sleep disorders by entering simple lifestyle and health details like sleep duration, stress level, and more.

Overview
This project uses Supervised Machine Learning to identify potential sleep disorders from health records.
The model is trained on a cleaned and preprocessed version of the Sleep Health dataset.
With an SVM model providing strong accuracy and generalization, the web interface is interactive and user-friendly for health analysis and predictions.

Author: Daiwik Korat, Aniket Mali

Features
Model Used: SVM (RBF Kernel)
Accuracy: ~92% on test data
Input Features:

Sleep Duration

Screen Time

Stress Level

Age

Gender

BMI Category

Systolic Blood Pressure

Diastolic Blood Pressure

Alcohol/Smoking Addiction

Functionality:

Real-time sleep disorder predictions

Handles raw user input & batch prediction (optional CSV upload)

Feature scaling with StandardScaler

Categorical encoding for multi-class features (BMI, Gender, etc.)

🛠️ Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/Sleep-Disorder-Prediction.git
cd Sleep-Disorder-Prediction
Set up virtual environment:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
Launch the Streamlit app:

bash
Copy
Edit
streamlit run app.py

Dataset
File: Updated_Sleep_Apnea_Sleep_health.csv

Source: Public sleep health datasets

Records: ~375+

Target: Disorder (CRSD,Restless Leg Syndrome, Sleep Apnea, Insomnia, Parasomnia)

Preprocessing:

Missing value handling

One-Hot Encoding

Feature scaling

BP column split into Systolic & Diastolic

Model Chosen
Algorithm: Support Vector Machine (SVM)
Why SVM?

Great performance on smaller datasets

Works well with clear margins of separation

Robust to high-dimensional features

Others Tested:

Logistic Regression

Random Forest

KNN

Gradient Boosting

📈 Performance Metrics
Accuracy: ~92%

Evaluation: Confusion Matrix, Classification Report

Train-Test Split: 80/20


Create a new branch: git checkout -b feature-name

Make your changes and commit: git commit -m "Add new feature"

Push to GitHub: git push origin feature-name

Open a Pull Request
