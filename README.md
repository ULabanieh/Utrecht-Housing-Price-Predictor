# 🚧 Project Status: In Progress
---

# Project Overview

The objective of this project is to develop and compare multiple machine learning models for predicting residential property prices in Utrecht using structured property-level data. The dataset consists of approximately 2,000 individual housing listings and includes quantitative features such as living area, lot size, number of bathrooms, construction year, energy efficiency, amenities, and valuation indicators.

Beyond building a single predictive model, the project is designed as a comparative study of different regression approaches, including linear models, tree-based methods, and a simple neural network. The aim is to evaluate how model complexity influences predictive performance and generalization.

A strong emphasis is placed on constructing a transparent and well-structured end-to-end data science pipeline. This includes data validation, feature selection, careful handling of potentially sensitive predictors (such as tax-based valuations), model training, performance evaluation using appropriate regression metrics, and critical reflection on model assumptions and limitations.

The focus is therefore not only on predictive accuracy, but also on methodological rigor, interpretability, and clear justification of modeling decisions.

---

## Data Loading and Initial Exploration

The project began by setting up the repository structure and initializing a Jupyter Notebook to support exploratory analysis and model development. The Utrecht housing dataset (`Utrechthousinghuge.csv`) was downloaded and loaded into a pandas DataFrame.

Link to the original dataset in Kaggle: https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset/data

Initial inspection steps were performed to understand the structure of the dataset, including the number of observations, available features, and data types. A preview of the data confirmed that each row represents an individual property listing and that the dataset is suitable for a supervised regression task.

---

## Data Cleaning

The dataset required minimal cleaning prior to modeling. A duplicate check was performed to ensure that no repeated property listings were present. It was observed that two listings shared the same `id` value despite having different property characteristics and prices. This suggests that the identifier does not uniquely represent a single property. As a result, the `id` column was removed and not used in the modeling process.

All features were already provided in numerical format, with categorical characteristics encoded as binary variables where applicable. This significantly reduced preprocessing complexity and allowed the focus to shift quickly toward feature preparation and model development.

Overall, the dataset was well-structured and suitable for direct use in supervised regression modeling.

---

## Feature Selection and Column Filtering

Before modeling, an initial feature selection step was carried out to remove columns that did not contribute meaningful predictive value or that could negatively affect model performance.

Identifier-related fields such as `id` were removed, as they function only as unique references and do not generalize across observations. The `zipcode` column was also excluded to avoid high-cardinality categorical data and potential location-based information leakage.

Additionally, the features `lot-len` and `lot-width` were dropped because they are highly correlated with `lot-area`, which already captures the overall size of the property’s lot. Removing these redundant features helps reduce multicollinearity and simplifies the modeling process without sacrificing relevant information.

The remaining features were retained based on their direct relevance to property characteristics, including size, amenities, construction year, energy efficiency, and valuation-related indicators. Together, these variables provide a balanced representation of physical and qualitative factors influencing housing prices.

---

## Column Renaming and Formatting

As part of the preprocessing stage, column names were standardized to improve readability and ensure compatibility with common Python data analysis and machine learning workflows. Columns containing hyphens were renamed using snake_case formatting to follow Python naming conventions and to allow for easier feature access and manipulation in code.

This step improves code clarity, reduces the risk of syntax-related errors, and ensures consistency across the project.

---

## Target Variable Definition

The `retailvalue` feature was selected as the target variable for this project. It represents the market value of each property and serves as the outcome the machine learning models aim to predict.

---

## Consideration of the `taxvalue` Feature

The `taxvalue` feature represents an official property valuation provided by public authorities and is typically based on market trends, location, and physical characteristics of the property. Due to its close relationship with actual market prices, this feature has the potential to significantly improve predictive performance.

However, because tax valuations are often derived using information similar to that used to determine market value, including this feature may introduce a form of target leakage, making the prediction task artificially easier and potentially obscuring the model’s ability to learn from intrinsic property characteristics.

To address this concern, two modeling approaches were adopted. One set of models was trained using all available features except `taxvalue`, focusing exclusively on physical and qualitative property attributes. A second set of models included `taxvalue` in order to assess the impact of official valuations on prediction performance.

Comparing these approaches enables a more transparent evaluation of model behavior and provides insight into how strongly tax assessments influence housing price predictions.
