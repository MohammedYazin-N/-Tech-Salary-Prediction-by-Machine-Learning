# -Tech-Salary-Prediction-by-Machine-Learning
 Predictive Modeling for finding salary in USD   for techies in  AI , Machine Learning and Data Science roles in the year 2020-2025  using Machine Learning Algorithms 
 
📄 Dataset Description
This dataset contains detailed information about tech industry job roles and salaries across various locations and companies. It is primarily used to predict salary in USD based on multiple employee and company-related attributes. The data spans different levels of experience, job titles, remote work ratios, and geographic locations, providing a rich context for building a machine learning salary prediction model.
🔢 Key Columns:
Column Name	Description
work_year	The year in which the salary was recorded.
experience_level	The seniority level of the employee (e.g., Entry-Level, Mid-Level, Senior, Executive).
employment_type	The type of employment contract (e.g., Full-time, Part-time, Freelance, Contract).
job_title	The specific role or job title held by the employee (e.g., Data Scientist, ML Engineer).
salary	The salary in the original local currency (not used in model training).
salary_currency	The currency in which the salary is paid (e.g., USD, EUR, INR).
salary_in_usd	The salary converted to USD (used as the target variable for prediction).
employee_residence	The country where the employee resides.
remote_ratio	The percentage of remote work (0 = on-site, 50 = hybrid, 100 = fully remote).
company_location	The location (country) of the company's headquarters.
company_size	The size of the company — Small (S), Medium (M), or Large (L).
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

🔧 Stage 1: Dataset Collection Imported the dataset using pandas

Inspected column names, data types, and missing values

Checked for duplicate records

Verified structure and feature relevance

🧼 Stage 2: Preprocessing & Cleaning 🔹 Handling Missing Values Missing values in alcohol_consumption (~60%) were filled with "Unknown" to retain behavioral information

No other columns had significant missing data

🔹 Categorical Encoding Identified object-type columns and applied one-hot encoding

drop_first=True used to prevent multicollinearity

🔹 Outlier Detection Used boxplots to visualize potential outliers in all numeric features

No removal was applied due to the clinical significance of outliers in healthcare (e.g., high blood pressure may indicate actual heart attack risk)

🔹 Feature Scaling Applied StandardScaler only to numeric features using a Pipeline with ColumnTransformer

This standardization ensures consistent scale for models sensitive to feature magnitude

🔹 Train/Test Split Split data into 80% training and 20% testing sets

Used stratified sampling to maintain class balance in heart_attack

✅ Current Project Status ✔ Preprocessing complete ⏳ Ready to proceed to Stage 3: Machine Learning Modeling & Evaluation
