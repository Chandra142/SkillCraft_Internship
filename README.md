# 🧠 Machine Learning Projects – Internship @ SkillCraft Technology

Welcome to my machine learning project repository! This repo contains two hands-on projects I completed as part of my internship at **SkillCraft Technology**. These projects demonstrate both supervised and unsupervised machine learning techniques using real-world datasets.

---

## 📌 Projects Overview

### 1️⃣ House Price Prediction (Linear Regression)
A supervised learning model that predicts house prices based on features such as square footage, number of bedrooms, and bathrooms.

**🔧 Tools & Libraries:**  
`Python`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `Google Colab`

**✅ Highlights:**
- Cleaned and preprocessed housing data
- Performed exploratory data analysis (EDA)
- Built and trained a linear regression model using `scikit-learn`
- Evaluated the model using R² Score and Mean Squared Error
- Visualized actual vs. predicted prices

📊 **Use Case:** Helps estimate house prices for property listings and real estate analysis.

---

### 2️⃣ Customer Segmentation (K-Means Clustering)
An unsupervised learning model that segments mall customers based on their annual income and spending score.

**🔧 Tools & Libraries:**  
`Python`, `Pandas`, `Seaborn`, `Matplotlib`, `Scikit-learn`

**✅ Highlights:**
- Selected relevant features and standardized data
- Applied K-Means clustering with 5 segments
- Mapped each cluster to intuitive customer types:
  - High Income - Low Spending
  - Low Income - High Spending
  - Average Income - Average Spending
  - etc.
- Visualized clusters with centroids using scatter plots

📊 **Use Case:** Useful for targeted marketing and customer behavior analysis.

---
3️⃣ Cat vs Dog Image Classifier (Support Vector Machine - SVM)  
A supervised classification model that identifies whether an uploaded image is a cat or a dog — and rejects unrelated images such as elephants, humans, or giraffes.

🔧 Tools & Libraries:  
Python, OpenCV, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab, Joblib

✅ Highlights:
- Loaded and preprocessed real-world image data  
- Resized and flattened images to create a feature matrix  
- Trained an SVM classifier to separate cats and dogs  
- Added confidence-based prediction logic:
  - ✅ Confident predictions: Cat or Dog
  - ❌ Rejected: Anything below the confidence threshold  
- Displayed visual results using matplotlib  
- Exported the final model as a `.pkl` file for reuse  

📊 Use Case: A lightweight image classifier that can serve as the backend for apps or web services where users upload photos of pets for classification.

---


## 📁 Folder Structure

```bash
.
├── house_price_prediction/
│   ├── linear_regression_model.ipynb
│   └── data/
├── customer_segmentation/
│   ├── kmeans_customer_segmentation.ipynb
│   └── Mall_Customers.csv
├── README.md
