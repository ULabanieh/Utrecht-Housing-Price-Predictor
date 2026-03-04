# Utrecht Housing Price Prediction - Code Walkthrough
**Step-by-Step Technical Explanation**

---

## 📚 Overview

This walkthrough explains every major code block in the Jupyter notebook. The key innovation is the **hybrid categorical encoding strategy** that optimizes preprocessing for each model type.

**Project Goal:** Predict Utrecht housing prices with 99%+ accuracy using physical features and location data.

---

## Section 1: Setup and Data Loading

### Step 1.1: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
```

**What each library does:**
- `pandas` → Data manipulation (DataFrames, CSV reading)
- `numpy` → Numerical operations (arrays, math functions)
- `matplotlib` → Basic plotting
- `seaborn` → Statistical visualizations
- `scipy.stats` → Statistical tests (skewness, distributions)

---

### Step 1.2: Set Random Seeds
```python
import random
random.seed(42)
np.random.seed(42)
```

**Purpose:** Ensures reproducible results  
**Why 42?** Arbitrary choice (common convention from "Hitchhiker's Guide to the Galaxy")  
**Effect:** Train-test splits, random forest sampling will be identical across runs

---

### Step 1.3: Load Dataset
```python
df = pd.read_csv('Utrechthousinghuge.csv')
df.head()
```

**What happens:**
- Reads CSV from disk into memory
- Creates DataFrame object with 2,000 rows × 15 columns
- `.head()` shows first 5 rows for inspection

---

### Step 1.4: Initial Data Inspection
```python
df.info()
df.describe()
```

**`df.info()` shows:**
- Column names and data types
- Non-null counts (detects missing values)
- Memory usage

**`df.describe()` shows:**
- Count, mean, std, min, quartiles, max
- Quick outlier detection

---

## Section 2: Data Quality Checks

### Step 2.1: Check for Duplicates
```python
df.duplicated().sum()
```

**Returns:** Count of duplicate rows  
**Expected:** 0 (no duplicates)  
**Why check?** Duplicates inflate sample size artificially

---

### Step 2.2: Check Missing Values
```python
df.isnull().sum()
```

**Returns:** Count of missing values per column  
**Expected:** 0 for all columns  
**Result:** 100% complete dataset ✅

---

### Step 2.3: Remove Redundant Features
```python
df = df.drop(columns=['id', 'lot-len', 'lot-width'])
```

**Why remove:**
- `id` → Not predictive (just identifier)
- `lot-len`, `lot-width` → Redundant (captured by `lot-area`)

<<<<<<< HEAD
**Why remove these?**
- `id` → Just a reference number, doesn't predict price
- `lot-len`, `lot-width` → Redundant (we have `lot-area`)
=======
**Result:** 15 columns → 12 columns
>>>>>>> 7bf51cb (final version of the project)

---

### Step 2.4: Rename Columns
```python
df = df.rename(columns={
    'lot-area': 'lot_area',
    'house-area': 'house_area',
    'garden-size': 'garden_size',
    'build-year': 'build_year',
    'taxvalue': 'tax_value',
    'retailvalue': 'retail_value'
})
```

**Why:** Python convention uses underscores (hyphens can cause errors)  
**Effect:** Cleaner code, prevents syntax issues

---

## Section 3: Neighborhood Engineering

### Step 3.1: Create Neighborhood Mapping
```python
df["zipcode"] = df["zipcode"].astype(str).str.strip()

zipcode_to_neighborhood = {
    "3528": "Kanaleneiland/Transwijk",
    "3525": "Tolsteeg",
    "3500": "Binnenstad",
    "3800": "Amersfoort"
}

df["neighborhood"] = df["zipcode"].map(zipcode_to_neighborhood)
```

**Line-by-line:**

**Line 1:** Convert zipcode to string and remove whitespace
- Ensures consistent format for mapping

**Lines 3-8:** Dictionary mapping postcodes to neighborhoods
- **Key design decision:** Aggregate ~200 postcodes into 4 areas
- Balances granularity vs. sample size

**Line 10:** Apply mapping
- Creates new `neighborhood` column
- Each property now has both zipcode and neighborhood

**Why this works:**
Dutch postcodes are hyper-local (10-20 houses). Even aggregated, they capture significant pricing variation:
- Binnenstad (city center): Premium pricing
- Amersfoort (suburban): Lower pricing
- Same house: €450k difference across neighborhoods!

---

### Step 3.2: Verify Distribution
```python
summary = df["neighborhood"].value_counts().reset_index()
summary.columns = ["neighborhood", "count"]
summary["percentage"] = (summary["count"] / summary["count"].sum() * 100)
print(summary)
```

**Output:**
```
Amersfoort:              639 (31.95%)
Tolsteeg:                508 (25.40%)
Kanaleneiland/Transwijk: 492 (24.60%)
Binnenstad:              361 (18.05%)
```

**What to check:**
- ✅ All neighborhoods represented (no missing mappings)
- ✅ Balanced distribution (18-32% each)
- ✅ Sufficient samples per category (>100 minimum)

---

## Section 4: Hybrid Encoding Strategy

### Step 4.1: Label Encoding (for Trees)
```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['neighborhood_label'] = label_encoder.fit_transform(df['neighborhood'])
```

**What happens:**
```
Amersfoort → 0
Binnenstad → 1
Kanaleneiland/Transwijk → 2
Tolsteeg → 3
```

**Why for trees?**
- Trees split on values: "if neighborhood == 2, then..."
- No ordering assumption (3 is not "greater" than 2)
- Simpler: 1 column vs 3 dummy variables
- Faster training and prediction

---

### Step 4.2: One-Hot Encoding (for Linear/NN)
```python
neighborhood_dummies = pd.get_dummies(df['neighborhood'], prefix='neighborhood', drop_first=True)
df_with_dummies = pd.concat([df, neighborhood_dummies], axis=1)
```

**What happens:**
Creates 3 binary columns (Amersfoort is baseline):
```
neighborhood_Binnenstad: [0 or 1]
neighborhood_Kanaleneiland/Transwijk: [0 or 1]
neighborhood_Tolsteeg: [0 or 1]
```

**Why `drop_first=True`?**
- Avoids multicollinearity (dummy variable trap)
- 4 categories need only 3 columns
- Baseline (Amersfoort) = all zeros

**Why for linear models?**

❌ **Problem with label encoding:**
```
Linear regression sees: 0, 1, 2, 3
Assumes: Tolsteeg (3) = 3 × Amersfoort (0)
Creates false mathematical relationship!
```

✅ **Solution with one-hot:**
```
Price = β₀ + β₁×house_area + β₂×(is_Binnenstad) + β₃×(is_Tolsteeg) + ...

Each neighborhood gets independent coefficient:
- Binnenstad: β₂ = +€180,000 (premium)
- Tolsteeg: β₃ = +€45,000 (premium)
- Amersfoort: baseline (no coefficient)
```

**No false ordering!**

---

### Step 4.3: Define Feature Lists
```python
base_features = ['lot_area', 'house_area', 'garden_size', 'build_year',
                 'bathrooms', 'energy_eff', 'monument']

# For tree models
features_tree_no_tax = base_features + ['neighborhood_label']  # 8 features
features_tree_with_tax = features_tree_no_tax + ['tax_value']  # 9 features

# For linear/NN models
neighborhood_cols = ['neighborhood_Binnenstad', 'neighborhood_Kanaleneiland/Transwijk', 
                     'neighborhood_Tolsteeg']
features_linear_no_tax = base_features + neighborhood_cols  # 10 features
features_linear_with_tax = features_linear_no_tax + ['tax_value']  # 11 features
```

**Summary:**

| Model Type | Encoding | Features (no tax) | Features (with tax) |
|------------|----------|-------------------|---------------------|
| Tree (RF, GB) | Label | 8 | 9 |
| Linear, NN | One-Hot | 10 | 11 |

---

## Section 5: Train-Test Split

### Step 5.1: Create Dataset Versions
```python
# For tree models (label-encoded)
X_tree_no_tax = df[features_tree_no_tax]
X_tree_with_tax = df[features_tree_with_tax]

# For linear/NN models (one-hot encoded)
X_linear_no_tax = df_with_dummies[features_linear_no_tax]
X_linear_with_tax = df_with_dummies[features_linear_with_tax]

y = df['retail_value']
```

**Creates 4 feature matrices:**
1. Tree WITHOUT tax (8 features)
2. Tree WITH tax (9 features)
3. Linear WITHOUT tax (10 features)
4. Linear WITH tax (11 features)

---

### Step 5.2: Perform Split
```python
from sklearn.model_selection import train_test_split

X_train_tree_no_tax, X_test_tree_no_tax, y_train, y_test = train_test_split(
    X_tree_no_tax, y, test_size=0.2, random_state=42
)

X_train_linear_no_tax, X_test_linear_no_tax, _, _ = train_test_split(
    X_linear_no_tax, y, test_size=0.2, random_state=42
)

# (Repeat for WITH tax scenarios)
```

**Parameters:**
- `test_size=0.2` → 20% test, 80% train
- `random_state=42` → Same split across all scenarios

**Result:**
- Training: 1,600 samples (80%)
- Test: 400 samples (20%)
- **Critical:** Same properties in train/test across ALL scenarios

---

### Step 5.3: Log Transform Target
```python
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
```

**What `np.log1p()` does:**
```python
log1p(x) = log(x + 1)
```

**Why +1?** Prevents log(0) = undefined

**Why log transform?**
- Original: Skewness = 0.615 (right-skewed)
- After log: Skewness = 0.053 (nearly symmetric)
- **91% reduction in skewness**

**Benefits:**
- More normal distribution → better for linear models
- Stabilizes variance across price ranges
- Predictions symmetric around mean

**Note:** Only used for Linear Regression and Neural Network

---

## Section 6: Feature Scaling

### Step 6.1: Scale for Linear/NN Models
```python
from sklearn.preprocessing import StandardScaler

scaler_linear_no_tax = StandardScaler()
X_train_scaled_linear_no_tax = scaler_linear_no_tax.fit_transform(X_train_linear_no_tax)
X_test_scaled_linear_no_tax = scaler_linear_no_tax.transform(X_test_linear_no_tax)
```

**What StandardScaler does:**
```python
z = (x - mean) / std
```
Transforms to: mean = 0, std = 1

**Example:**
```
house_area before: [50, 100, 150, 200, 250] m²
house_area after:  [-1.4, -0.7, 0, 0.7, 1.4] (standardized)
```

**Why scale?**
- house_area: 50-250
- build_year: 1920-2018
- Different scales → features not comparable
- Scaling puts all features on same footing

**Critical pattern:**
```python
scaler.fit(X_train)        # Learn mean/std from TRAINING only
X_train = scaler.transform(X_train)  # Apply to training
X_test = scaler.transform(X_test)    # Apply to test (same mean/std)
```

**Never:** `scaler.fit(X_test)` → Data leakage!

---

### Step 6.2: Tree Models Don't Need Scaling
```python
# Use X_train_tree_no_tax directly (no scaling)
```

**Why?**
- Trees split on thresholds: "if house_area > 150, then..."
- Threshold adapts to scale automatically
- Scaling adds no benefit (wastes computation)

---

## Section 7: Linear Regression

### Step 7.1: Train Model
```python
from sklearn.linear_model import LinearRegression

lr_no_tax = LinearRegression()
lr_no_tax.fit(X_train_scaled_linear_no_tax, y_train_log)
```

**What `.fit()` does:**
- Solves: `β = (X'X)⁻¹X'y`
- Finds coefficients minimizing squared error
- No iterative training (closed-form solution)

**Model equation:**
```
log(price) = β₀ + β₁×house_area + β₂×lot_area + ... + β₉×neighborhood_Tolsteeg
```

---

### Step 7.2: Make Predictions
```python
y_test_pred_log = lr_no_tax.predict(X_test_scaled_linear_no_tax)
y_test_pred = np.expm1(y_test_pred_log)
```

**Line 1:** Predict in log scale  
**Line 2:** Back-transform to euros using `expm1(x) = e^x - 1`

**Why back-transform?**
Model predicts log(price), we need actual price in euros

---

### Step 7.3: Evaluate
```python
from sklearn.metrics import mean_squared_error, r2_score

test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)
```

**RMSE (Root Mean Squared Error):**
```
RMSE = √(mean((actual - predicted)²))
```
- In euros (interpretable)
- Penalizes large errors more

**R² (R-squared):**
```
R² = 1 - (sum of squared errors / total variance)
```
- 0 to 1 (higher better)
- 0.97 = explains 97% of variance

**Expected Results:**
- RMSE ≈ €35,800
- R² ≈ 0.9712

---

## Section 8: Random Forest

### Step 8.1: Train Model
```python
from sklearn.ensemble import RandomForestRegressor

rf_no_tax = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_no_tax.fit(X_train_tree_no_tax, y_train)
```

**Hyperparameters explained:**

**`n_estimators=100`:**
- Number of trees in forest
- More trees = better performance, but diminishing returns
- 100 is good balance

**`max_depth=20`:**
- Maximum depth of each tree
- Controls model complexity
- Deeper = more complex patterns, but risk overfitting

**`min_samples_split=5`:**
- Minimum samples required to split a node
- Prevents splitting on very small groups

**`min_samples_leaf=2`:**
- Minimum samples at leaf node
- Prevents leaves with single samples (overfitting)

**`n_jobs=-1`:**
- Use all CPU cores for parallel training
- Speeds up training significantly

**How Random Forest works:**
1. Build 100 independent decision trees
2. Each tree trained on bootstrap sample (random subset with replacement)
3. Each split considers random subset of features
4. Final prediction = average of all 100 trees

---

### Step 8.2: Feature Importance
```python
feature_importance = pd.DataFrame({
    'Feature': features_tree_no_tax,
    'Importance': rf_no_tax.feature_importances_
}).sort_values('Importance', ascending=False)
```

**What it shows:**
```
house_area:         75.2%
neighborhood_label: 19.8%
build_year:         3.1%
lot_area:           0.9%
...
```

**How calculated:**
- Measure error reduction from each feature across all trees
- Sum across all splits and trees
- Normalize to percentages

**Interpretation:**
- house_area dominates (75%) but more balanced than without location
- neighborhood_label critical (20%)
- Other features contribute minimally

---

## Section 9: Gradient Boosting

### Step 9.1: Train Model
```python
from sklearn.ensemble import GradientBoostingRegressor

gb_no_tax = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)

gb_no_tax.fit(X_train_tree_no_tax, y_train)
```

**Hyperparameters explained:**

**`learning_rate=0.1`:**
- Shrinkage parameter (how much each tree contributes)
- Lower = more conservative (prevents overfitting)
- 0.1 is common default

**`max_depth=5`:**
- Shallow trees (vs RF's depth=20)
- Boosting works better with "weak learners"
- Prevents individual trees from overfitting

**`subsample=0.8`:**
- Use 80% of data for each tree
- Adds stochasticity (like bagging)
- Improves generalization

**How Gradient Boosting differs from Random Forest:**

**Random Forest:**
```
Tree 1 → Prediction A
Tree 2 → Prediction B
Tree 3 → Prediction C
...
Final = Average(A, B, C, ...)
```

**Gradient Boosting:**
```
Iteration 1: Predict mean price → Error₁
Iteration 2: Build tree to predict Error₁ → Error₂
Iteration 3: Build tree to predict Error₂ → Error₃
...
Final = Initial + 0.1×(Tree₁ + Tree₂ + Tree₃ + ...)
```

**Key difference:** Sequential error correction vs parallel averaging

---

### Step 9.2: Why It Wins

**Optimal bias-variance tradeoff:**
- Individual trees: High bias (shallow), low variance
- Boosting: Reduces bias through sequential learning
- Subsampling: Reduces variance through randomness

**Result:**
- Training R² = 0.9980
- Test R² = 0.9965
- Gap = 0.15% (excellent generalization!)

---

## Section 10: Neural Network

### Step 10.1: Define Architecture
```python
import torch
import torch.nn as nn

class HousingPriceNN(nn.Module):
    def __init__(self, input_size):
        super(HousingPriceNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.network(x)
```

**Architecture breakdown:**

**Layer 1:** 10 inputs → 16 neurons
- `nn.Linear(10, 16)` → Weighted sum
- `nn.ReLU()` → Activation (keeps positive, zeros negative)
- `nn.BatchNorm1d(16)` → Normalizes activations
- `nn.Dropout(0.2)` → Randomly turns off 20% of neurons

**Layer 2:** 16 → 8 neurons
- Same structure as Layer 1

**Layer 3:** 8 → 1 output
- Final prediction (price)

**Total parameters:** ~300 weights + biases

---

### Step 10.2: Training Loop
```python
model = HousingPriceNN(input_size=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Training process:**
1. **Forward pass:** Input → Model → Predictions
2. **Calculate loss:** How wrong are predictions? (MSE)
3. **Backward pass:** Calculate gradients (how to improve)
4. **Update weights:** Adjust parameters to reduce loss

**Repeats** for ~500 epochs until early stopping

---

### Step 10.3: Why It Underperforms

**Problem:** Insufficient data
- Neural networks need 10,000+ samples
- Only have 1,600 training samples
- Can't learn complex patterns reliably

**Evidence:**
- High bias (training R² = 0.936)
- Model underfits (can't fully learn patterns)
- Heavy regularization prevents overfitting but limits capacity

**Conclusion:** Use tree models for small tabular datasets

---

## Section 11: Multicollinearity Analysis (VIF)

### Step 11.1: Calculate VIF
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_numeric = X_train_linear_with_tax.values.astype(float)

vif_data = pd.DataFrame()
vif_data["Feature"] = features_linear_with_tax
vif_data["VIF"] = [variance_inflation_factor(X_train_numeric, i) 
                   for i in range(len(features_linear_with_tax))]
```

**What VIF measures:**
```
VIF = 1 / (1 - R²)
```
Where R² is from regressing that feature on all others

**Interpretation:**
- VIF = 1: No correlation with other features
- VIF = 5: Moderate correlation
- VIF = 10: High correlation (problematic)
- VIF > 100: Severe multicollinearity
- VIF > 1000: Catastrophic (model unstable)

**Results WITH tax_value:**
```
tax_value:  VIF = 5,528,356  (catastrophic!)
house_area: VIF = 4,117,212  (catastrophic!)
```

**Why?**
```
tax_value ≈ f(house_area, lot_area, build_year, ...)
```
Tax assessments are DERIVED from physical features!

**Conclusion:** Including tax_value creates circular prediction with no benefit

---

## Section 12: Visualization Examples

### Step 12.1: Model Comparison Bar Chart
```python
models = ['Gradient Boosting', 'Random Forest', 'Linear Regression', 'Neural Network']
rmse_values = [12500, 14200, 35800, 52000]

plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
plt.ylabel('Test RMSE (€)')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

**Creates:** Bar chart showing RMSE by model  
**Insight:** Visual comparison makes GB's superiority clear

---

### Step 12.2: Feature Importance Plot
```python
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Gradient Boosting')
plt.tight_layout()
```

**Creates:** Horizontal bar chart of feature importance  
**Insight:** house_area and neighborhood dominate

---

### Step 12.3: Actual vs Predicted Scatter
```python
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_test_pred_gb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (€)')
plt.ylabel('Predicted Price (€)')
plt.legend()
```

**Creates:** Scatter plot with diagonal reference line  
**How to read:**
- Points near red line = good predictions
- Points above line = over-predicted
- Points below line = under-predicted
- Tight cluster = consistent accuracy

---

### Step 12.4: Residual Distribution
```python
residuals = y_test - y_test_pred_gb

plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residual (€)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
```

**What to look for:**
- Centered at 0? (no systematic bias)
- Symmetric? (no over/under-prediction tendency)
- Normal shape? (validates assumptions)

---

## 🎯 Key Patterns to Remember

### Pattern 1: Fit on Train, Transform Both
```python
# ✅ CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ WRONG (data leakage)
scaler.fit(X_test)
```

### Pattern 2: Different Data for Different Models
```python
# Linear/NN: Use scaled, one-hot encoded, log-transformed
lr.fit(X_train_scaled_linear_no_tax, y_train_log)

# Trees: Use unscaled, label-encoded, original target
rf.fit(X_train_tree_no_tax, y_train)
```

### Pattern 3: Back-Transform Predictions
```python
# Linear/NN predictions are in log scale
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)  # Convert back to euros
```

---

## 💡 Summary

This project demonstrates:
1. **Location matters:** 20% importance from neighborhood
2. **Hybrid encoding works:** Tailored preprocessing improves results
3. **Gradient Boosting wins:** Sequential error correction optimal for this task
4. **Neural networks need more data:** Underperform on small tabular datasets
5. **Tax value is redundant:** Multicollinearity without benefit

**Final Result:** 99.65% R² (€12,500 RMSE) on €791k properties

---
