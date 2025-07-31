# -Tech-Salary-Prediction-by-Machine-Learning
 Predictive Modeling for finding salary in USD   for techies in  AI , Machine Learning and Data Science roles in the year 2020-2025  using Machine Learning Algorithms 
 
📄 Dataset Description

This dataset contains detailed information about tech industry job roles and salaries across various locations and companies. It is primarily used to predict salary in USD based on multiple employee and company-related attributes. The data spans different levels of experience, job titles, remote work ratios, and geographic locations, providing a rich context for building a machine learning salary prediction model.

🔢 Key Columns:

| Column Name             | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **work\_year**          | The year in which the salary was recorded.                                               |
| **experience\_level**   | The seniority level of the employee (e.g., Entry-Level, Mid-Level, Senior, Executive).   |
| **employment\_type**    | The type of employment contract (e.g., Full-time, Part-time, Freelance, Contract).       |
| **job\_title**          | The specific role or job title held by the employee (e.g., Data Scientist, ML Engineer). |
| **salary**              | The salary in the original local currency (not used in model training).                  |
| **salary\_currency**    | The currency in which the salary is paid (e.g., USD, EUR, INR).                          |
| **salary\_in\_usd**     | The salary converted to USD (used as the target variable for prediction).                |
| **employee\_residence** | The country where the employee resides.                                                  |
| **remote\_ratio**       | The percentage of remote work (0 = on-site, 50 = hybrid, 100 = fully remote).            |
| **company\_location**   | The location (country) of the company's headquarters.                                    |
| **company\_size**       | The size of the company — Small (S), Medium (M), or Large (L).                           |


🎯 Target Variable:

    salary_in_usd – The annual salary of the employee converted to US dollars, which is the value the machine learning model aims to predict.

💡 Objective of the Project:

The goal is to build a machine learning regression model that can accurately predict the salary in USD based on a combination of categorical and numerical features such as job title, experience level, company size, remote work percentage, and location.

🛠Tasks 
 Clean and preprocess a real-world  dataset

Handle missing values and outliers appropriately

Encode categorical variables and scale numeric features

Split the dataset into training and testing subsets

Prepare the data for machine learning modeling 

📁 Dataset

Source: https://deepdatalake.com/index.php

Name: The AI^J ML^J Data Science Salary

Rows:88584

Columns: 11 (including the target salary_in_usd)

Type: Regression 

🧼 Stage 1: Dataset Collection Imported the dataset using pandas

Inspected column names, data types, and missing values

Checked for duplicate records

Verified structure and feature relevance

🧼 Stage 2: Preprocessing & Cleaning 🔹 

🔹Removing Duplicates
    Identified and removed duplicate rows to avoid biased model training.

🔹 Outlier Detection & Treatment

    Detected outliers in the salary_in_usd (target) column using the IQR method:

        Computed Q1 (25th percentile) and Q3 (75th percentile)

        Defined outlier bounds:
        Lower Bound = Q1 - 1.5 × IQR
        Upper Bound = Q3 + 1.5 × IQR

    Filtered out extreme salary values outside these bounds to prevent distortion in the model.

 🔹Skewness Correction

           Analyzed the skewness of the target variable (salary_in_usd), which showed a strong right-skew.

           Applied log transformation (log1p) to reduce skewness and improve model performance.

🔹  Data Type Conversion

           Ensured appropriate data types for each column (e.g., converting categorical columns to object or category).
 
🔹  Categorical Feature Encoding
      Converted categorical variables like job_title, company_size, etc., into numerical format using  One-Hot Encoding as required.


🔹  Feature Scaling Applied StandardScaler  to training Data

This standardization ensures consistent scale for models sensitive to feature magnitude

🔹  Train/Test Split Split data into 80% training and 20% testing sets


✅ Current Project Status ✔ Preprocessing complete ⏳

🚀  Ready to proceed to Stage 3: Machine Learning Modeling & Evaluation🚀 
