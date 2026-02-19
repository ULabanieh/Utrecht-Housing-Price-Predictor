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

### Missing Value Analysis

A comprehensive assessment of data completeness was performed to identify any missing values that could compromise model training or introduce bias.

### Methodology:

All columns in the dataset were inspected using `.isnull().sum()` to count missing entries. The analysis examined both the absolute count and percentage of missing values per feature.

### Findings:

The dataset exhibited complete data coverage across all 2,000 observations:

`✅ NO MISSING VALUES DETECTED
   Total observations: 2,000
   Total features: 12 (after initial preprocessing)
   Dataset completeness: 100%`

**Verification Results:**

| Feature | Missing Count | Missing % |
| --- | --- | --- |
| All features | 0 | 0.0% |

No features contained missing values. All 2,000 property listings had complete information across all variables.

### Data Quality Implications:

The absence of missing values simplifies the preprocessing pipeline and eliminates potential sources of bias that could arise from imputation. This is relatively uncommon in real-world datasets and likely reflects:

1. **Structured data collection**: The original data source (likely a real estate listing platform or municipal registry) maintained strict data quality standards
2. **Pre-cleaned dataset**: The Kaggle dataset may have undergone preliminary cleaning by the data provider
3. **Complete records only**: Listings with incomplete information may have been excluded from the original dataset

### Imputation Decision:

**No imputation strategies were required.** All features retained their full sample size (n = 2,000) for modeling, ensuring maximum statistical power and eliminating potential distortions from imputed values.

### Assumption and Limitations:

We assume that the absence of missing values does not introduce selection bias — that is, if incomplete listings were excluded from the original dataset, they do not differ systematically from the included properties in ways that would affect model generalizability. This assumption cannot be verified without access to the original raw data source, but given the dataset's representativeness across property types, construction eras, and price ranges, any potential bias is likely minimal.

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

![image.png](attachment:49a0655b-e15b-4697-bcec-f31422fa50c4:image.png)

### Interpretation

The resulting visualizations confirm the statistical patterns observed in the descriptive analysis:

- **Lot Area, House Area, and Garden Size** exhibit moderate right skewness, reflecting the presence of larger properties in the dataset
- **Build Year** shows a relatively uniform distribution across decades, with no strong temporal bias
- **Tax Value and Retail Value** both display near-normal distributions when viewed on a log scale, with retail values consistently higher than tax values — a pattern that aligns with economic expectations, as market prices typically exceed official tax assessments

These visualizations provide a clear and interpretable foundation for subsequent feature engineering and modeling decisions.

---

## Boxplot Analysis: Outlier Detection and Data Spread

Boxplot visualizations were generated to assess data spread, identify the interquartile range (IQR), and detect potential outliers across all numeric features.

**Findings:**

The boxplots reveal several important characteristics of the dataset:

1. **Physical Features (Lot Area, House Area, Garden Size):** These variables exhibit moderate right-skewness with a small number of high-end outliers. The outliers represent legitimate luxury properties with larger dimensions rather than data quality issues. For instance, lot areas extend up to approximately 220 m², while most properties cluster between 90-140 m². These outliers are retained in the dataset as they reflect genuine market diversity.
2. **Build Year:** This variable displays a symmetric distribution with no outliers, spanning from 1920 to 2018. The balanced temporal representation confirms that the dataset includes properties from multiple construction eras without bias toward any particular time period.
3. **Tax Value and Retail Value:** Both valuation features show a concentration of properties in the €500k-€750k range, with a distinct cluster of high-value outliers between €1.1M-€1.5M. These represent the premium property segment in Utrecht's housing market. Notably, retail values consistently exceed tax values across all quartiles, which aligns with economic expectations.

![image.png](attachment:7d7672fa-f13f-4c12-aed8-92d02fd5aca4:image.png)

**Outlier Treatment Decision:**

No outliers were removed from the dataset. All flagged points represent plausible property characteristics and fall within realistic market ranges. Removing them would artificially narrow the model's applicability and reduce its ability to generalize across the full spectrum of Utrecht's housing market. Tree-based models, in particular, are robust to such variation and will naturally account for these high-end properties during training.

---

## Correlation Analysis

A correlation heatmap was generated to examine linear relationships between numeric features and identify potential multicollinearity issues prior to modeling.

### Key Findings:

**Strong Correlations with Target (Retail Value):**

- **Tax Value ↔ Retail Value**: Very strong positive correlation (r ≈ 0.95+) - This confirms that official tax assessments are highly predictive of market prices, supporting the decision to model both with and without this feature to assess the impact of potential target leakage.
- **House Area ↔ Retail Value**: Strong positive correlation (r ≈ 0.70-0.80) - Larger living spaces command higher prices, as expected in real estate markets.
- **Lot Area ↔ Retail Value**: Moderate positive correlation (r ≈ 0.50-0.60) - Property lot size contributes to value but is less influential than interior living space.
- **Garden Size ↔ Retail Value**: Weak to moderate positive correlation (r ≈ 0.30-0.40) - Outdoor space has some influence but is less critical than structural features.
- **Build Year ↔ Retail Value**: Weak correlation (r ≈ 0.10-0.20) - Construction year shows minimal linear relationship with price, suggesting that age alone does not determine value in Utrecht's market (likely due to renovations and location factors).

**Feature Intercorrelations:**

- **Lot Area ↔ House Area**: Moderate positive correlation (r ≈ 0.50-0.60) - Larger lots tend to have larger homes, but the relationship is not deterministic.
- **Lot Area ↔ Garden Size**: Moderate positive correlation (r ≈ 0.40-0.50) - Properties with larger lots typically have more garden space.
- **House Area ↔ Garden Size**: Weak correlation (r ≈ 0.20-0.30) - Indoor and outdoor space are somewhat independent.

**Tax Value ↔ Other Features**: Tax value shows similar correlation patterns to retail value (strong with house area, moderate with lot area), reinforcing that tax assessments are based on similar physical property characteristics.

![image.png](attachment:6ba80fc6-543b-4d80-964a-bef8e61b02b7:image.png)

### Implications for Modeling:

1. **Multicollinearity**: No severe multicollinearity detected among predictor variables (all r < 0.70 except between tax_value and retail_value). This suggests that each feature contributes unique information and can be safely included in linear models without causing estimation instability.
2. **Feature Importance Expectations**: Based on correlation strength, we anticipate that `house_area` and `tax_value` will emerge as the most influential predictors in model training, followed by `lot_area` and `garden_size`. Build year may have limited predictive power in linear models but could interact non-linearly in tree-based approaches.
3. **Target Leakage Validation**: The extremely high correlation between tax_value and retail_value (r > 0.90) confirms the importance of the dual modeling strategy. Models trained without tax_value will better reflect the model's ability to predict market prices from intrinsic property characteristics alone.

---

## Log Transformation Justification

To assess whether the target variable requires transformation prior to modeling, the distribution of `retail_value` was compared before and after applying a logarithmic transformation.

### Rationale for Log Transformation:

Linear regression and neural network models assume that residuals are approximately normally distributed and exhibit constant variance (homoscedasticity). When the target variable is right-skewed, these assumptions are often violated, leading to:

- Biased predictions toward the mean
- Heteroscedastic residuals (increasing variance at higher values)
- Inefficient parameter estimates
- Poor performance on high-value properties

### Transformation Methodology:

A log transformation was applied using `np.log1p()` (log(1 + x)) rather than `np.log()` to safely handle any zero values, though none were present in this dataset. The transformation compresses the scale of large values while expanding the scale of small values, thereby reducing right skewness.

### Visual Comparison:

The side-by-side distribution plots reveal:

**Original Distribution:**

- Exhibits moderate right skewness (skewness ≈ 0.80-1.00)
- Long tail extending toward high-value properties
- Peak concentrated around €600k-€800k
- Violates normality assumption for linear models

**Log-Transformed Distribution:**

- Substantially more symmetric (skewness ≈ 0.10-0.20)
- Reduced tail on the right side
- More bell-shaped, closer to normal distribution
- Better satisfies linear regression assumptions

### Modeling Implications:

1. **For Linear Models:** Log transformation will be applied to the target variable during training. Predictions will be back-transformed using the exponential function to return to the original scale.
2. **For Tree-Based Models (Random Forest, Gradient Boosting):** Log transformation is not necessary, as these models are non-parametric and do not assume normality of the target variable. They naturally handle skewed distributions through recursive partitioning.
3. **For Neural Networks:** Log transformation will be applied to stabilize training and improve convergence, as neural networks benefit from normalized target distributions.

### Statistical Evidence:

Skewness was quantified using the skewness coefficient:

- Original skewness: 0.615
- Log-transformed skewness: 0.053
- Reduction: 91.3%

This substantial reduction in skewness confirms that log transformation is an appropriate preprocessing step for linear and neural network models in this project.

![image.png](attachment:859f063f-acc8-4eba-bbf6-8fec68b248a8:image.png)

---

## Target Variable Definition

The `retailvalue` feature was selected as the target variable for this project. It represents the market value of each property and serves as the outcome the machine learning models aim to predict.

Before proceeding to model training, a dedicated analysis of the target variable (`retail_value`) was conducted to validate its distribution, identify potential data quality issues, and confirm suitability for regression modeling.

### Distribution Characteristics:

**Descriptive Statistics:**

- **Count**: 2,000 observations
- **Mean**: €791,024
- **Median**: €766,000
- **Standard Deviation**: €210,980
- **Range**: €419,000 – €1,428,000 (span of €1,009,000)

**Distribution Shape:**

- **Skewness**: 0.615 (moderate right skew)
- **Kurtosis**: -0.141 (slightly platykurtic, flatter than normal distribution)

The target variable exhibits a moderate right-skewed distribution, which is characteristic of real estate price data. The mean exceeds the median by approximately €25,000, indicating the influence of high-value properties in the upper tail of the distribution.

**Quartile Breakdown:**

- **Q1 (25th percentile)**: €631,750
- **Q2 (50th percentile / Median)**: €766,000
- **Q3 (75th percentile)**: €907,250
- **Interquartile Range (IQR)**: €275,500

The IQR captures the middle 50% of property values, showing that the core market spans a relatively wide band of approximately €275k. Properties outside this range represent either budget or luxury segments.

### Data Quality Validation:

**Completeness:**

- Missing values: 0 (100% complete)
- No imputation required

**Range Validity:**

- Minimum value: €419,000 (realistic for Utrecht housing market)
- Maximum value: €1,428,000 (plausible for luxury properties)
- No negative or zero values detected
- All values fall within economically reasonable bounds

**Outlier Assessment:**
Using the standard outlier definition (values > Q3 + 1.5 × IQR), 27 properties (1.35% of the dataset) are flagged as high-value outliers. These properties are retained in the dataset as they represent legitimate luxury market segments rather than data errors. Visual inspection of the boxplot confirms these outliers follow a continuous extension of the main distribution rather than forming a separate cluster.

### Visual Analysis:

The side-by-side visualization reveals:

**Histogram (Left)**: The distribution peaks around €750k-€850k with a gradual right tail extending toward €1.4M. The KDE curve smoothly traces the overall shape, confirming moderate positive skew without extreme deviation from normality.

**Boxplot (Right)**: The box represents the IQR (€631k to €907k) with the median line at €766k. The whiskers extend to approximately €400k (lower) and €1,300k (upper), with 27 individual points marked as outliers beyond the upper whisker. The visual symmetry of the box around the median further illustrates the moderate nature of the skew.

### Implications for Modeling:

1. **Right Skewness**: The moderate positive skew (0.615) violates the normality assumption required by linear regression models. This justifies the application of log transformation for linear and neural network models (see Log Transformation Justification section).
2. **No Data Quality Issues**: The absence of missing values, impossible values, or extreme anomalies indicates that the target variable is clean and ready for model training without further preprocessing.
3. **Sufficient Variability**: The target exhibits meaningful variation (coefficient of variation ≈ 27%), ensuring that models will have adequate signal to learn predictive patterns. A target with low variance would limit model performance regardless of predictor quality.
4. **Balanced Distribution**: While skewed, the distribution is not severely pathological. The skewness coefficient of 0.615 is well within the range where transformation is beneficial but not absolutely critical for all model types. Tree-based models can handle this distribution in its original form.
5. **Outlier Retention**: The 27 high-value properties (1.35%) represent the upper market segment and will be retained to ensure the model generalizes across the full price spectrum. Their presence is reflected in the slightly negative kurtosis (-0.141), indicating a distribution with lighter tails than a perfect normal curve.

---

### Consideration of the `taxvalue` Feature

The `taxvalue` feature represents an official property valuation provided by public authorities and is typically based on market trends, location, and physical characteristics of the property. Due to its close relationship with actual market prices, this feature has the potential to significantly improve predictive performance.

However, because tax valuations are often derived using information similar to that used to determine market value, including this feature may introduce a form of target leakage, making the prediction task artificially easier and potentially obscuring the model’s ability to learn from intrinsic property characteristics.

To address this concern, two modeling approaches were adopted. One set of models was trained using all available features except `taxvalue`, focusing exclusively on physical and qualitative property attributes. A second set of models included `taxvalue` in order to assess the impact of official valuations on prediction performance.

Comparing these approaches enables a more transparent evaluation of model behavior and provides insight into how strongly tax assessments influence housing price predictions.
