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

# 🔄 PROJECT WORKFLOW 🔄

## 🧼 Stage 1: Dataset Understanding and Cleaning 

🔹 Imported the dataset using pandas

🔹 Inspected column names, data types, and missing values

🔹 Checked for duplicate records

🔹Removing Duplicates
    Identified and removed duplicate rows to avoid biased model training.

🔹 Verified structure and feature relevance

## 🧼  Stage 2 : Exploratory Data Analysis (EDA)
🔹 Able to explain "Job Title Distribution" using  Count Plot.

🔹 Storytells about "Salary vs. Experience Level" using Box plot

🔹 Visualised "Average Salary by Company Location"using Bar plot

🔹Understands "Salary vs. Company Size" using Violin Plot

🔹Able to explain  "Salary by Remote Work Ratio"using Box plot

🔹Viewing "Trend Over Time " using Line plot


## 🧼 Stage 3  : Preprocessing  

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

## 🧼 Stage 4: Model Building and Evaluation

To achieve reliable predictions, I experimented with a diverse set of regression algorithms ranging from simple linear models to advanced ensemble methods. This approach ensures both **baseline comparison** and **performance optimization**.

### 📌 Models Implemented

* **Linear Regression** → Baseline model for reference.
* **Decision Tree Regressor** → Captures non-linear relationships.
* **Random Forest Regressor** → Ensemble of trees to improve accuracy and reduce overfitting.
* **Support Vector Regressor (SVR)** → Captures complex patterns using kernel functions.
* **HistGradientBoosting Regressor (HGBR)** → Fast and memory-efficient gradient boosting, ideal for large datasets.

### 📊 Evaluation Strategy

All models were trained and compared using key regression metrics:

* **R² Score**
* **Mean Squared Error (MSE)**
* **Mean Absolute Error (MAE)**

This helped in identifying the **best-performing algorithm** for the salary prediction task.

## 🔎 Key Observations

*📌Best Model:

The HistGradientBoostingRegressor achieved the highest R² score (0.2883) and the lowest MSE, making it the best-performing model among those tested.

It indicates that this model explains about 28.8% of the variance in salaries.

*📌Close Competition:

Decision Tree, Linear Regression, and Random Forest models all performed similarly, with R² scores around 0.27–0.28.

Their error values (MSE, MAE) were also in the same range, suggesting that none of them significantly outperformed the others.

*📌Weak Performer:

The KNeighbors Regressor performed the worst, with an R² of only 0.19, indicating poor generalization capability for this dataset.

*📌 Interpretation

An R² around 0.28 suggests that the models capture only a moderate portion of salary variability.

This is acceptable in real-world datasets, especially when predicting salaries where many hidden or unmeasured factors (e.g., individual skills, negotiation, company policies) affect the outcome.


## 🧼 Stage 5: Hyperparameter Tuning with Randomized Search CV

After building baseline models, the next step was to **optimize model performance** by fine-tuning hyperparameters. Instead of manually guessing parameter values, I used **RandomizedSearchCV**, which efficiently samples from a wide range of hyperparameter combinations and performs cross-validation to identify the best configuration.

### 📌 Why Randomized Search?

* **Efficiency** → Unlike Grid Search, which tests all combinations, Randomized Search samples a fixed number of combinations, reducing computation time.
* **Exploration** → Covers a wider parameter space, increasing the chance of finding a near-optimal set of hyperparameters.
* **Cross-Validation** → Ensures robust evaluation by averaging results across folds.

### ⚙️ Applied Process

* Selected the **HistGradientBoosting Regressor (HGBR)**, as it gave the best performance among baseline models.

* Defined a **parameter grid** including:

  * `max_iter` → Number of boosting iterations
  * `learning_rate` → Step size for updates
  * `max_leaf_nodes` → Controls tree complexity
  * `min_samples_leaf` → Minimum samples per leaf
  * `l2_regularization` → Regularization strength

* Performed **RandomizedSearchCV** over these parameters with cross-validation.

* Identified the **best hyperparameters**:

### 📊 Results

* **Best CV R² Score**: ` 0.2997` (improved compared to baseline HGBR).
* Retrained the model with the best parameters and evaluated it on the **test dataset**.

### Evaluate on Test Data 
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Test R² Score:", r2_score(y_test, y_pred))

**Test R² Score** : 0.2940
✅ Key Takeaway

Hyperparameter tuning with RandomizedSearchCV significantly enhanced the model’s generalization. The tuned HGBR achieved improved performance both in cross-validation and on unseen test data, making it a strong candidate for the final predictive model.

## 🧼 Stage 6: Prediction

After fine-tuning and finalizing the best model, the next step was to **use the trained model for predictions on new unseen data**.

### 📌 How it works

* The new dataset (`new_data`) must have the **same features** as the training dataset (except the target column `salary_in_usd`).
* The finalized **best model** (in this case, the tuned HistGradientBoosting Regressor) is used to generate salary predictions.
* This step ensures the pipeline is **ready for real-world application**, where the model will predict salaries for new job postings or employee profiles.


### ✅ Key Takeaway

This stage demonstrates the **practical utility** of the project. Once trained and optimized, the model is not just an academic exercise—it can now be applied to **real business problems**, such as forecasting salaries based on job roles, experience, company size, location, and work setup.


## 🔚 Conclusion

This project successfully demonstrates the **end-to-end process of building a salary prediction model** using machine learning. Starting from **data preprocessing** and **outlier handling**, progressing through **model building** with diverse algorithms, and further refining through **hyperparameter tuning**, the workflow highlights the importance of experimentation and systematic evaluation in data science.

Although the tuned model achieved a **moderate R² score (\~0.29)**, the project provides a strong foundation for future improvements such as:

* Incorporating **feature engineering** to capture deeper relationships in the data.
* Exploring **advanced models** (e.g., XGBoost, CatBoost, or Neural Networks).
* Applying **cross-validation strategies** to further enhance generalization.

Most importantly, the project demonstrates how a trained machine learning model can be applied to **real-world predictions on new, unseen data**—turning raw information into actionable insights.

This journey reflects the **practical challenges of predictive modeling** and showcases how systematic steps can lead to a working solution, even if the initial performance leaves room for enhancement.




