# 🚧 Project Status: In Progress
---

# Project Overview

The objective of this project is to develop and compare multiple machine learning models for predicting residential property prices in Utrecht using structured property-level data. The dataset consists of approximately 2,000 individual housing listings and includes quantitative features such as living area, lot size, number of bathrooms, construction year, energy efficiency, amenities, and valuation indicators.

Beyond building a single predictive model, the project is designed as a comparative study of different regression approaches, including linear models, tree-based methods, and a simple neural network. The aim is to evaluate how model complexity influences predictive performance and generalization.

A strong emphasis is placed on constructing a transparent and well-structured end-to-end data science pipeline. This includes data validation, feature selection, careful handling of potentially sensitive predictors (such as tax-based valuations), model training, performance evaluation using appropriate regression metrics, and critical reflection on model assumptions and limitations.

The focus is therefore not only on predictive accuracy, but also on methodological rigor, interpretability, and clear justification of modeling decisions.

---

## Reproducibility

To ensure consistent and reproducible results across model training runs, random seeds were fixed for Python, NumPy, and PyTorch. Additionally, all stochastic algorithms such as train-test splitting and Random Forest were initialized with a fixed random state.

---

## Data Loading and Initial Exploration

The project began by setting up the repository structure and initializing a Jupyter Notebook to support exploratory analysis and model development. The Utrecht housing dataset (`Utrechthousinghuge.csv`) was downloaded and loaded into a pandas DataFrame.

Link to the original dataset in Kaggle: https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset/data

Initial inspection steps were performed to understand the structure of the dataset, including the number of observations, available features, and data types. A preview of the data confirmed that each row represents an individual property listing and that the dataset is suitable for a supervised regression task.

---

## Data Preprocessing

The dataset required minimal cleaning prior to modeling. A duplicate check was performed to ensure that no repeated property listings were present. It was observed that a small number of listings shared the same `id` value despite having different property characteristics and prices. This suggests that the identifier does not uniquely represent a single property. As a result, the `id` column was removed and not used in the modeling process.

All features were already provided in numerical format, with categorical characteristics encoded as binary variables where applicable. This significantly reduced preprocessing complexity and allowed the focus to shift quickly toward feature preparation and model development.

Overall, the dataset was well-structured and suitable for direct use in supervised regression modeling.

---

### Feature Selection and Column Filtering

Before modeling, an initial feature selection step was carried out to remove columns that did not contribute meaningful predictive value or that could negatively affect model performance.

Identifier-related fields such as `id` were removed, as they function only as unique references and do not generalize across observations. The `zipcode` column was also excluded to avoid high-cardinality categorical data and potential location-based information leakage.

Additionally, the features `lot-len` and `lot-width` were dropped because they are highly correlated with `lot-area`, which already captures the overall size of the property’s lot. Removing these redundant features helps reduce multicollinearity and simplifies the modeling process without sacrificing relevant information.

The remaining features were retained based on their direct relevance to property characteristics, including size, amenities, construction year, energy efficiency, and valuation-related indicators. Together, these variables provide a balanced representation of physical and qualitative factors influencing housing prices.

---

### Column Renaming and Formatting

As part of the preprocessing stage, column names were standardized to improve readability and ensure compatibility with common Python data analysis and machine learning workflows. Columns containing hyphens were renamed using snake_case formatting to follow Python naming conventions and to allow for easier feature access and manipulation in code.

This step improves code clarity, reduces the risk of syntax-related errors, and ensures consistency across the project.

---

### Feature Type Identification

All features in the dataset are numerical in format. However, two variables — `energy_eff` and `monument` — represent categorical property characteristics encoded as binary values (0 = No, 1 = Yes).

These columns were inspected using the `.unique()` method and confirmed to contain only valid binary values:

- `energy_eff`: [0, 1]
- `monument`: [0, 1]

Since they are already properly encoded, no additional transformation (e.g., one-hot encoding) was required.

---

## Descriptive Statistics and Distribution Analysis

The dataset contains 2,000 observations with no missing values across the inspected numerical features. Overall, the statistical summary indicates a well-structured and realistic housing dataset.

- **Build Year:**
    
    The `build_year` variable ranges from 1920 to 2018, with a mean of 1969 and near-zero skewness. The symmetric distribution suggests a balanced representation of properties across different construction periods.
    
- **Lot Area and House Area:**
    
    The average lot size is 115 m², while the average living area is 140 m². Both variables exhibit moderate positive skewness (≈1), indicating the presence of larger properties that extend the upper tail of the distribution. This pattern is expected in real estate data and does not suggest data irregularities.
    
- **Garden Size:**
    
    Garden size shows moderate right skewness, reflecting that while most properties have modest outdoor space, a smaller number have significantly larger gardens.
    
- **Bathrooms:**
    
    The distribution is slightly positively skewed, likely due to most properties having one bathroom and fewer properties having multiple bathrooms.
    
- **Tax Value and Retail Value:**
    
    The mean tax value (€651,715) and mean retail value (€791,024) are closely aligned, with retail values consistently higher. This is economically reasonable, as market listing prices often exceed official tax assessments. Both variables exhibit moderate positive skewness, which is typical for housing price distributions.
    

Overall, no extreme or implausible values were detected. The observed variation appears to reflect natural market diversity rather than data quality issues.

---

## Distribution Visualization of Numeric Features

To understand the underlying structure and variability of the dataset, distribution plots were generated for the six primary numeric features: lot area, house area, garden size, build year, tax value, and retail value.

Each variable was visualized using a histogram with an overlaid kernel density estimate (KDE) curve to provide both frequency counts and a smoothed representation of the distribution shape.

### Visualization Design Decisions

The plots were arranged in a 2×3 grid layout to facilitate direct comparison across features. A sophisticated color scheme was applied throughout: slate blue (`#5B7C99`) for histogram bars and muted crimson (`#C1666B`) for KDE curves. This color palette was chosen to create a professional, academic aesthetic while maintaining clear visual differentiation between histogram and density elements.

**Handling High-Value Features:**

The `tax_value` and `retail_value` features exhibit a wide numeric range (from approximately €300,000 to over €1,000,000), which posed challenges for readability on a linear scale. To address this, a logarithmic scale was applied to the x-axis for these two variables.

However, logarithmic scaling introduced a secondary issue: matplotlib's default tick formatting displayed values in scientific notation (e.g., 4 × 10⁵ instead of €400,000), which reduced interpretability for non-technical audiences.

To resolve this, custom tick positions and labels were explicitly defined using `FixedLocator` and `FixedFormatter`. Tick values were set at €300k, €500k, €700k, and €1M, with labels formatted using abbreviated notation (k for thousands, M for millions) to improve readability without sacrificing precision. Minor ticks were suppressed using `NullLocator` to prevent residual scientific notation from appearing on the axis.

**Color Implementation:**

The KDE line color required special handling due to seaborn's `histplot` parameter structure. Rather than using the `line_kws` or `kde_kws` parameters (which do not accept color arguments directly), the KDE line styling was applied post-creation by iterating through the axis line objects and manually setting their color and linewidth properties. This approach ensured consistent application of the muted crimson color across all KDE curves.

**Additional Styling:**

To enhance clarity and reduce visual clutter, the following refinements were applied:

- Gridlines were added to the y-axis only, using dashed lines with reduced opacity
- Top and right spines were removed from each subplot
- A centered figure title was added to provide context for the entire grid
- Individual subplot titles were cleaned and labeled with human-readable feature names and appropriate units (m² for area measurements, € for monetary values)

<img width="1589" height="1014" alt="image" src="https://github.com/user-attachments/assets/86955edf-6ebe-45f6-86aa-e6fd23b7f895" />

### Interpretation

The resulting visualizations confirm the statistical patterns observed in the descriptive analysis:

- **Lot Area, House Area, and Garden Size** exhibit moderate right skewness, reflecting the presence of larger properties in the dataset
- **Build Year** shows a relatively uniform distribution across decades, with no strong temporal bias
- **Tax Value and Retail Value** both display near-normal distributions when viewed on a log scale, with retail values consistently higher than tax values — a pattern that aligns with economic expectations, as market prices typically exceed official tax assessments

These visualizations provide a clear and interpretable foundation for subsequent feature engineering and modeling decisions.

---

## Target Variable Definition

The `retailvalue` feature was selected as the target variable for this project. It represents the market value of each property and serves as the outcome the machine learning models aim to predict.

---

### Consideration of the `taxvalue` Feature

The `taxvalue` feature represents an official property valuation provided by public authorities and is typically based on market trends, location, and physical characteristics of the property. Due to its close relationship with actual market prices, this feature has the potential to significantly improve predictive performance.

However, because tax valuations are often derived using information similar to that used to determine market value, including this feature may introduce a form of target leakage, making the prediction task artificially easier and potentially obscuring the model’s ability to learn from intrinsic property characteristics.

To address this concern, two modeling approaches were adopted. One set of models was trained using all available features except `taxvalue`, focusing exclusively on physical and qualitative property attributes. A second set of models included `taxvalue` in order to assess the impact of official valuations on prediction performance.

Comparing these approaches enables a more transparent evaluation of model behavior and provides insight into how strongly tax assessments influence housing price predictions.
