# [Phân tích dữ liệu thông minh] FinalProject
# 🎬 Movie Success Prediction Project

## 📌 Project Overview

This project, **"Movie Success Prediction"**, was developed as a final assignment for the course *Intelligent Data Analysis* at the University of Science. The project applies machine learning techniques to analyze and forecast the commercial potential and audience appeal of movies.

As the film industry becomes increasingly competitive, production companies and investors require tools that can help them make data-driven decisions, optimize resources, and improve the chances of a project's success. Our goal is to build predictive models capable of accurately estimating **revenue** and **popularity** based on movie features such as budget, genre, release time, and audience feedback.

---

## 🎯 Objectives

- Build regression models to predict a movie’s **revenue** and **popularity**.
- Identify and analyze key factors influencing film success.
- Provide actionable insights and strategic recommendations for producers and investors.
- Develop a user-friendly web app that allows users to input film details and receive predictions and suggestions for improvement.

---

## 📊 Dataset

- Source: [Millions of Movies - Kaggle](https://www.kaggle.com/datasets/erdemaydin/millions-of-movies)
- Records: 884,422 movies
- Features: 23 original columns
- Final processed dataset: 89,418 records × 17 features (after cleaning, transformation, and feature engineering)

---

## 🧪 Methodology

### 📌 Preprocessing

- Removed missing, zero, and outlier values
- Extracted and encoded relevant features (e.g., log-transformation, label encoding, datetime processing)
- Created derived features (e.g., genre rank, runtime ratio)

### 🤖 Machine Learning Models

- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **CatBoost Regressor**

### 🧠 Model Evaluation

- Metrics: R², MAE, RMSE
- Achieved:
  - **R² ~ 0.82** for revenue prediction
  - **R² ~ 0.78** for popularity prediction

### 🔍 Feature Importance

- Used **SHAP (SHapley Additive exPlanations)** to interpret model predictions and assess feature contributions.

---

## 🌐 Web Application

A Streamlit-based web app was developed with two main functionalities:

1. **Revenue Prediction** – Allows users to input basic movie info and receive predicted revenue.
2. **Improvement Suggestions** – Recommends changes to optimize movie performance using a similarity-based recommendation system.

---

## 🚀 Future Work

- Integrate additional data sources, especially from streaming platforms (e.g., Netflix, Prime).
- Expand the recommendation system to include actor/crew suggestions based on collaborative filtering.
- Deploy the app online to support independent filmmakers and small studios.

---

## 👨‍💻 Team Information – GGWP

| Student ID | Full Name              | Role         |
|------------|------------------------|--------------|
| 22120153   | Trần Duy Khang         | Team Leader  |
| 22120121   | Lê Viết Hưng           | Member       |
| 22120138   | Nguyễn Thành Huy       | Member       |
| 22120113   | Nguyễn Việt Hoàng      | Member       |
| 22120154   | Trịnh Hoàng Khang      | Member       |

---

## 📁 Repository Structure

