# Car-Dheko---Used-Car-Price-Prediction
End-to-end ML project predicting used car prices with cleaned multi-city data. Features preprocessing, feature engineering, model tuning, and a Streamlit app that provides instant, accurate price predictions for customers and dealerships.e best-performing model deployed for real-time use.
---

### 🎯 Objectives
- Build a reliable ML model that predicts used car prices.
- Clean, preprocess, and merge unstructured city-wise datasets.
- Perform EDA to identify key price-driving features.
- Compare multiple regression models and tune hyperparameters.
- Deploy the best-performing model using Streamlit.

---

### 🧩 Project Workflow

#### 1. **Data Preprocessing**
- Import and combine multiple city Excel files.
- Handle missing values and incorrect formats.
- Encode categorical data using One-Hot/Label Encoding.
- Scale numerical features and remove outliers.
- Add new columns like `car_age` for better predictions.

#### 2. **Exploratory Data Analysis (EDA)**
- Performed descriptive statistics to understand data distribution.
- Visualized patterns using histograms, scatter plots, and heatmaps.
- Identified strong correlations between key features and price.

#### 3. **Model Development**
- Algorithms used: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
- Evaluated models using **MAE**, **MSE**, and **R²** metrics.
- Selected the best model (Random Forest) for deployment.

#### 4. **Model Deployment**
- Integrated the trained model into a **Streamlit app**.
- Users can input car details and get instant price predictions.

---

### ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit  
- **Tools:** Jupyter Notebook, Git, VS Code  
- **Domain:** Automotive, Data Science, Machine Learning  

---
streamlit run app/streamlit_app.py
📊 Evaluation Metrics
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
User feedback for app usability

🧠 Key Learnings
End-to-end data science workflow from raw data to deployment.
Importance of data preprocessing and feature engineering.
Model selection, hyperparameter tuning, and performance optimization.
Building and deploying ML models using Streamlit.

📄 Results
Developed an accurate model for predicting used car prices.
Created interactive Streamlit web app for real-time predictions.
Achieved strong performance with low MAE and high R².

🏷️ Tags
Machine Learning · Data Science · Price Prediction · Regression · Streamlit · Python · EDA · Model Deployment

---

Would you like me to make a **shorter README version (under 1 page)** — ideal for GitHub display w
