# Utrecht Housing Price Prediction - Code Walkthrough
**Step-by-Step Explanation of the Jupyter Notebook**

---

## 📚 Section 1: Setup and Data Loading

### Step 1.1: Import Core Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
```

**What this does:**
- `pandas` → Data manipulation (loading CSV, creating dataframes)
- `numpy` → Numerical operations (arrays, mathematical functions)
- `matplotlib` → Plotting and visualization
- `seaborn` → Advanced statistical plots
- `scipy.stats` → Statistical functions (skewness calculation)

**Why we need it:** These are the foundation for data analysis in Python.

---

### Step 1.2: Set Random Seeds for Reproducibility
```python
import random
random.seed(42)
np.random.seed(42)
```

**What this does:**
- Sets a fixed "random" seed so results are identical every time you run the code
- The number 42 is arbitrary (could be any number)

**Why we need it:** Makes results reproducible - important for scientific work and debugging.

---

### Step 1.3: Load the Dataset
```python
df = pd.read_csv('Utrechthousinghuge.csv')
```

**What this does:**
- Reads the CSV file from disk
- Stores it in a pandas DataFrame called `df`

**What's a DataFrame?** Think of it like an Excel spreadsheet - rows are properties, columns are features.

---

### Step 1.4: Preview the Data
```python
df.head()
```

**What this does:**
- Shows the first 5 rows of the dataset
- Lets you see what features (columns) exist and their data types

**What you're checking:** Does the data look reasonable? Are there obvious problems?

---

### Step 1.5: Check Data Structure
```python
df.info()
```

**What this does:**
- Shows number of rows and columns
- Lists all column names and their data types
- Shows how many non-null (non-missing) values each column has

**Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2000 entries, 0 to 1999
Data columns (total 16 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   id           2000 non-null   int64  
 1   zipcode      2000 non-null   int64  
 2   lot-len      2000 non-null   float64
 3   lot-width    2000 non-null   float64
 4   lot-area     2000 non-null   float64
 5   house-area   2000 non-null   float64
 6   garden-size  2000 non-null   float64
 7   balcony      2000 non-null   int64  
 8   x-coor       2000 non-null   int64  
 9   y-coor       2000 non-null   int64  
 10  buildyear    2000 non-null   int64  
 11  bathrooms    2000 non-null   int64  
 12  taxvalue     2000 non-null   int64  
 13  retailvalue  2000 non-null   int64  
 14  energy-eff   2000 non-null   int64  
 15  monument     2000 non-null   int64  
dtypes: float64(5), int64(11)
memory usage: 250.1 KB
```

---

### Step 1.6: Statistical Summary
```python
df[['lot_area', 'house_area', 'garden_size', 'build_year', 'tax_value', 'retail_value']].describe()
```

**What this does:**
- Calculates statistics for numeric columns: count, mean, std, min, 25%, 50%, 75%, max
- Helps identify outliers and understand data distribution

**What to look for:**
- Are min/max values reasonable?
- Is the scale consistent across features?
- Are there extreme outliers?

**Output:**
	lot_area	house_area	garden_size	build_year	tax_value	retail_value
count	2,000	2,000	2,000	2,000	2,000	2,000
mean	115	140	35	1,969	651,715	791,024
std	34	42	24	26	182,927	210,980
min	50	68	5	1,920	310,000	419,000
25%	89	111	14	1,947	521,000	631,750
50%	110	135	32	1,969	633,000	766,000
75%	138	166	53	1,992	759,250	907,250
max	216	248	116	2,018	1,162,000	1,428,000


---

## 🧹 Section 2: Data Cleaning

### Step 2.1: Check for Duplicates
```python
df.duplicated().sum()
```

**What this does:**
- Counts how many rows are exact duplicates
- Returns a number (e.g., 0 means no duplicates)

**Why it matters:** Duplicate data can bias model training.

---

### Step 2.2: Check for Missing Values
```python
df.isnull().sum()
```

**What this does:**
- For each column, counts how many values are missing (NaN/None)
- Returns a Series showing missing count per column

**Example output:**
```
id              0
retailvalue     0
lot-area       0
```

**In this project:** 0 missing values! Clean dataset.

---

### Step 2.3: Remove Unnecessary Columns
```python
df = df.drop(columns=['id', 'zipcode', 'lot-len', 'lot-width'])
```

**What this does:**
- Removes 4 columns from the dataset
- Creates a new DataFrame (overwrites `df`)

**Why remove these?**
- `id` → Just a reference number, doesn't predict price
- `zipcode` → Too many unique values, causes problems
- `lot-len`, `lot-width` → Redundant (we have `lot-area`)

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

**What this does:**
- Changes column names from `lot-area` to `lot_area`
- Follows Python naming convention (underscores instead of hyphens)

**Why?** Hyphens can cause syntax errors in Python (they're minus signs!).

---

### Step 2.5: Verify Binary Columns
```python
print(df['energy_eff'].unique())
print(df['monument'].unique())
```

**What this does:**
- Shows all unique values in these columns
- Expected: [0, 1] for binary features

**Output:**
```
[0 1]
[0 1]
```

**Interpretation:**
- `energy_eff`: 0 = not certified, 1 = certified
- `monument`: 0 = regular property, 1 = designated monument

---

## 📊 Section 3: Exploratory Data Analysis (EDA)

### Step 3.1: Target Variable Analysis
```python
print(f"Count: {df['retail_value'].count()}")
print(f"Mean: €{df['retail_value'].mean():,.0f}")
print(f"Median: €{df['retail_value'].median():,.0f}")
print(f"Std Dev: €{df['retail_value'].std():,.0f}")
```

**What this does:**
- Calculates basic statistics for the target variable (price)
- Formats output with commas and euro sign

**Example output:**
```
Count: 2000
Mean: €791,024
Median: €766,000
Std Dev: €210,980
```

**What to notice:** Mean > Median suggests right-skewed distribution (more expensive properties pull the average up).

---

### Step 3.2: Calculate Skewness
```python
skewness = skew(df['retail_value'])
print(f"Skewness: {skewness:.3f}")
```

**What this does:**
- Measures asymmetry of the distribution
- Positive skew = tail on right side (high values)

**Output:** `Skewness: 0.615`

**Interpretation:**
- 0 = perfectly symmetric
- 0.5 to 1 = moderate positive skew
- > 1 = highly skewed

---

### Step 3.3: Distribution Histogram
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(df['retail_value'], bins=30, color='#5B7C99', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Retail Value (€)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Retail Value')
```

**What this does - Line by line:**

**Line 1:** Creates a figure with 2 side-by-side plots (1 row, 2 columns)
- `figsize=(14, 5)` → 14 inches wide, 5 inches tall
- `ax1, ax2` → References to the two plot areas

**Line 3:** Creates histogram on first plot
- `bins=30` → Divides data into 30 bars
- `color='#5B7C99'` → Slate blue color
- `alpha=0.7` → 70% opacity (slight transparency)
- `edgecolor='black'` → Black borders on bars

**Lines 4-6:** Add labels and title
- Makes the plot readable and professional

**Why histograms?** Show the shape of the data distribution visually.

---

### Step 3.4: Distribution with KDE
```python
sns.histplot(df['retail_value'], bins=30, kde=True, color='#5B7C99', 
             line_kws={'linewidth': 2, 'color': '#C1666B'})
```

**What this does:**
- `histplot` → Creates histogram
- `kde=True` → Adds a smooth curve (Kernel Density Estimate)
- `line_kws` → Customizes the KDE line (width and color)

**What's KDE?** A smooth curve that shows the distribution shape without discrete bars.

---

### Step 3.5: Boxplot for Outliers
```python
plt.boxplot(df['retail_value'], vert=False)
plt.xlabel('Retail Value (€)')
plt.title('Boxplot of Retail Value')
```

**What this does:**
- Creates a horizontal boxplot
- Shows: median (line in box), quartiles (box edges), outliers (dots)

**How to read a boxplot:**
```
    |----[====|====]----•
    ^    ^    ^    ^    ^
   Min  Q1   Q2   Q3  Outliers
            (Median)
```

**What you see:**
- Box = middle 50% of data (Q1 to Q3)
- Whiskers = typical range
- Dots = outliers (unusual values)

---

### Step 3.6: Correlation Heatmap
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
```

**What this does - Step by step:**

**Line 1:** Finds all numeric columns (ignores text columns)

**Line 2:** Calculates correlation between every pair of columns
- Returns a matrix (table) of correlation values

**Line 4-5:** Creates the heatmap
- `annot=True` → Show correlation numbers on cells
- `fmt='.3f'` → Format numbers to 3 decimal places
- `cmap='RdBu_r'` → Color scheme (red-white-blue)
- `center=0` → White color at 0 correlation
- `square=True` → Make cells square-shaped
- `linewidths=0.5` → Thin borders between cells

**How to read correlations:**
- +1.0 = Perfect positive correlation (both increase together)
- 0.0 = No correlation
- -1.0 = Perfect negative correlation (one increases, other decreases)

**Example:** If house_area ↔ retail_value = 0.97, bigger houses strongly predict higher prices.

---

## 🔬 Section 4: Log Transformation

### Step 4.1: Compare Original vs Log-Transformed
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Original distribution
ax1.hist(df['retail_value'], bins=30, color='#5B7C99', alpha=0.8)
ax1.set_title('Original Distribution')

# Log-transformed distribution
retail_value_log = np.log1p(df['retail_value'])
ax2.hist(retail_value_log, bins=30, color='#5B7C99', alpha=0.8)
ax2.set_title('Log-Transformed Distribution')
```

**What this does:**

**`np.log1p()`:**
- Applies natural logarithm: log(x + 1)
- The "+1" prevents errors if x=0

**Why log transform?**
- Right-skewed data (long tail on right) becomes more symmetric
- Helps linear models make better predictions
- Makes large values (€1M) and small values (€400k) more comparable

**Visual effect:**
- Original: Tail stretches to the right
- Log-transformed: More bell-shaped (normal distribution)

---

### Step 4.2: Calculate Skewness Reduction
```python
original_skew = skew(df['retail_value'])
log_skew = skew(retail_value_log)
reduction = ((original_skew - log_skew) / original_skew) * 100

print(f"Original skewness: {original_skew:.3f}")
print(f"Log-transformed skewness: {log_skew:.3f}")
print(f"Reduction: {reduction:.1f}%")
```

**What this does:**
- Calculates skewness before and after transformation
- Computes percentage reduction

**Example output:**
```
Original skewness: 0.615
Log-transformed skewness: 0.053
Reduction: 91.3%
```

**Interpretation:** Transformation reduced skewness by 91%, making data much more symmetric.

---

## 🔀 Section 5: Train-Test Split

### Step 5.1: Define Features and Target
```python
# Features WITHOUT tax_value
features_without_tax = ['lot_area', 'house_area', 'garden_size', 
                         'build_year', 'bathrooms', 'energy_eff', 'monument']

# Features WITH tax_value
features_with_tax = features_without_tax + ['tax_value']

# Target variable
target = 'retail_value'
```

**What this does:**
- Creates lists of column names to use as inputs (features)
- Separates features from target (what we're predicting)

**Two scenarios:**
1. Without tax → 7 features
2. With tax → 8 features

**Why two scenarios?** Test if tax_value helps or creates data leakage.

---

### Step 5.2: Split the Data
```python
from sklearn.model_selection import train_test_split

X_no_tax = df[features_without_tax]
X_with_tax = df[features_with_tax]
y = df[target]

X_train_no_tax, X_test_no_tax, y_train, y_test = train_test_split(
    X_no_tax, y, test_size=0.2, random_state=42
)

X_train_with_tax, X_test_with_tax, _, _ = train_test_split(
    X_with_tax, y, test_size=0.2, random_state=42
)
```

**What this does - Line by line:**

**Lines 3-5:** Create feature matrices (X) and target vector (y)
- `X` = input features (the predictors)
- `y` = target variable (what we want to predict)

**Lines 7-9:** Split WITHOUT tax_value
- `test_size=0.2` → 20% for testing, 80% for training
- `random_state=42` → Same split every time

**Result:**
- `X_train_no_tax` = 1,600 rows for training
- `X_test_no_tax` = 400 rows for testing
- `y_train` = 1,600 prices for training
- `y_test` = 400 prices for testing

**Lines 11-13:** Split WITH tax_value
- Uses same `random_state` so same properties end up in train/test
- `_, _` ignores the duplicate y_train, y_test (already have them)

**Why split?**
- Training set → Model learns patterns
- Test set → Evaluate if model generalizes to new data

---

### Step 5.3: Verify Split
```python
print(f"Training set size: {len(X_train_no_tax)}")
print(f"Test set size: {len(X_test_no_tax)}")
print(f"Total: {len(X_train_no_tax) + len(X_test_no_tax)}")
```

**What this does:**
- Checks that split worked correctly
- Verifies no data was lost

**Expected output:**
```
Training set size: 1600
Test set size: 400
Total: 2000
```

---

## 📏 Section 6: Feature Scaling

### Step 6.1: Scale Features for Linear Models
```python
from sklearn.preprocessing import StandardScaler

scaler_no_tax = StandardScaler()
X_train_scaled_no_tax = scaler_no_tax.fit_transform(X_train_no_tax)
X_test_scaled_no_tax = scaler_no_tax.transform(X_test_no_tax)
```

**What this does - Step by step:**

**Line 3:** Create a scaler object
- StandardScaler transforms features to mean=0, std=1

**Line 4:** Fit and transform training data
- `fit` → Learns mean and std from training data
- `transform` → Applies the transformation

**Line 5:** Transform test data
- Uses the SAME mean/std learned from training
- **Critical:** Never fit on test data (causes data leakage)

**Formula:** `z = (x - mean) / std`

**Example:**
- Original: house_area ranges 50-250 m²
- Scaled: ranges approximately -2 to +2

**Why scale?**
- Features on different scales (year: 1920-2018, bathrooms: 1-3)
- Linear models work better when features are comparable
- Prevents large-scale features from dominating

---

### Step 6.2: Transform Target Variable
```python
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
```

**What this does:**
- Applies log transformation to target for linear/neural models
- Reduces skewness (makes more normal)

**Why?**
- Linear regression assumes normally distributed target
- Helps model predict expensive and cheap houses equally well

**Note:** Tree models (Random Forest, Gradient Boosting) will use original `y_train`, not `y_train_log`.

---

## 🤖 Section 7: Linear Regression

### Step 7.1: Train the Model
```python
from sklearn.linear_model import LinearRegression

lr_no_tax = LinearRegression()
lr_no_tax.fit(X_train_scaled_no_tax, y_train_log)
```

**What this does:**

**Line 3:** Create empty Linear Regression model
- No parameters needed (uses default OLS)

**Line 4:** Train the model
- `.fit()` → Learn coefficients from training data
- Takes: scaled features and log-transformed target
- Returns: trained model (stored in `lr_no_tax`)

**What happens internally:**
- Model finds best line: ŷ = β₀ + β₁x₁ + β₂x₂ + ...
- Minimizes squared errors between predictions and actual values

---

### Step 7.2: Make Predictions
```python
y_train_pred_log_no_tax = lr_no_tax.predict(X_train_scaled_no_tax)
y_test_pred_log_no_tax = lr_no_tax.predict(X_test_scaled_no_tax)
```

**What this does:**
- Uses trained model to predict prices
- Returns predictions in log scale

**Line 1:** Predict on training data
- Used to check if model learned the training patterns

**Line 2:** Predict on test data
- Used to evaluate if model generalizes to new data

---

### Step 7.3: Back-Transform Predictions
```python
y_train_pred_no_tax = np.expm1(y_train_pred_log_no_tax)
y_test_pred_no_tax = np.expm1(y_test_pred_log_no_tax)
```

**What this does:**
- Converts predictions from log scale back to euros
- `expm1()` is the inverse of `log1p()`

**Why?**
- Model predicted in log scale
- We need euros for evaluation and interpretation

**Example:**
- Log prediction: 13.5
- Back-transformed: exp(13.5) - 1 ≈ €729,000

---

### Step 7.4: Calculate Metrics
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred_no_tax))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_no_tax))
train_r2 = r2_score(y_train, y_train_pred_no_tax)
test_r2 = r2_score(y_test, y_test_pred_no_tax)
```

**What this does:**

**RMSE (Root Mean Squared Error):**
- Formula: √(mean of squared errors)
- Measures average prediction error in euros
- Lower is better

**R² (R-squared):**
- Formula: 1 - (sum of squared errors / total variance)
- Measures % of variance explained
- Range: 0 to 1 (higher is better)
- 0.96 = model explains 96% of price variation

**Why both metrics?**
- RMSE → Tells you error in euros (interpretable)
- R² → Tells you model quality (standardized)

---

### Step 7.5: Print Results
```python
print(f"Training RMSE: €{train_rmse:,.0f}")
print(f"Test RMSE: €{test_rmse:,.0f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
```

**Example output:**
```
Training RMSE: €39,085
Test RMSE: €42,161
Training R²: 0.9654
Test R²: 0.9614
```

**What to look for:**
- Test RMSE close to train RMSE → Good generalization
- Test R² close to train R² → Not overfitting
- High R² (>0.90) → Strong model

---

## 🌲 Section 8: Random Forest

### Step 8.1: Train Random Forest
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

rf_no_tax.fit(X_train_no_tax, y_train)
```

**What this does - Parameter by parameter:**

**`n_estimators=100`:**
- Build 100 decision trees
- More trees = better but slower

**`max_depth=20`:**
- Each tree can be up to 20 levels deep
- Deeper = can learn more complex patterns

**`min_samples_split=5`:**
- Need at least 5 samples to split a node
- Prevents splitting on tiny groups

**`min_samples_leaf=2`:**
- Each leaf must have at least 2 samples
- Prevents overfitting to individual properties

**`random_state=42`:**
- Reproducible results

**`n_jobs=-1`:**
- Use all CPU cores (parallel training)

**Line 10:** Train the model
- Uses ORIGINAL features (no scaling needed for trees)
- Uses ORIGINAL target (no log transform needed)

**What happens:**
1. Creates 100 trees, each on random subset of data
2. Each tree makes independent predictions
3. Final prediction = average of all trees

---

### Step 8.2: Get Feature Importance
```python
feature_importance = pd.DataFrame({
    'Feature': features_without_tax,
    'Importance': rf_no_tax.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)
```

**What this does:**
- Extracts importance scores from trained model
- Higher importance = feature contributes more to predictions

**How importance is calculated:**
- Based on how much each feature reduces prediction error
- Across all 100 trees and all splits

**Example output:**
```
    Feature     Importance
0   house_area      0.953
1   build_year      0.038
2   lot_area        0.004
```

**Interpretation:** house_area contributes 95.3% to predictions!

---

## 🚀 Section 9: Gradient Boosting

### Step 9.1: Train Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingRegressor

gb_no_tax = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

gb_no_tax.fit(X_train_no_tax, y_train)
```

**What this does - Key parameters:**

**`learning_rate=0.1`:**
- How much each tree contributes
- Lower = more careful (prevents overfitting)
- Higher = faster but risky

**`max_depth=5`:**
- Shallow trees (vs RF's depth=20)
- Boosting works better with weak learners

**`subsample=0.8`:**
- Use 80% of data for each tree
- Adds randomness to prevent overfitting

**Training process (simplified):**
1. Start with average price
2. Build tree to predict errors
3. Add tree to ensemble (scaled by learning_rate)
4. Repeat 100 times

**Key difference from Random Forest:**
- RF: Trees independent (parallel)
- GB: Trees sequential (each fixes previous errors)

---

## 🧠 Section 10: Neural Network

### Step 10.1: Define Network Architecture
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

**What this does - Layer by layer:**

**`nn.Linear(input_size, 16)`:**
- Fully connected layer: input → 16 neurons
- For 7 features: creates 7×16 = 112 weights

**`nn.ReLU()`:**
- Activation function: keeps positive values, zeros out negatives
- Introduces non-linearity

**`nn.BatchNorm1d(16)`:**
- Normalizes activations across batch
- Stabilizes training

**`nn.Dropout(0.2)`:**
- Randomly turns off 20% of neurons during training
- Prevents overfitting

**Architecture flow:**
```
Input (7 features)
    ↓
Dense layer: 16 neurons
    ↓
ReLU activation
    ↓
Batch normalization
    ↓
Dropout (20%)
    ↓
Dense layer: 8 neurons
    ↓
ReLU activation
    ↓
Batch normalization
    ↓
Dropout (20%)
    ↓
Output: 1 neuron (price prediction)
```

---

### Step 10.2: Training Loop
```python
model = HousingPriceNN(input_size=7)
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

**What this does - Step by step:**

**Line 1:** Create model instance

**Line 2:** Define loss function
- MSELoss = Mean Squared Error
- Measures how wrong predictions are

**Line 3:** Define optimizer
- Adam = adaptive learning rate algorithm
- `lr=0.01` = learning rate (how big are update steps)

**Training loop (Lines 5-12):**

**For each epoch:**
1. **Forward pass** (line 7): Input → Model → Predictions
2. **Calculate loss** (line 8): How wrong are predictions?
3. **Zero gradients** (line 10): Clear previous gradients
4. **Backward pass** (line 11): Calculate gradients (how to improve)
5. **Update weights** (line 12): Adjust model parameters

**Simplified analogy:**
- Like practicing free throws
- Each epoch = one practice session
- Loss = measure of how many you missed
- Backward pass = analyze what went wrong
- Update weights = adjust your technique

---

### Step 10.3: Early Stopping
```python
best_loss = float('inf')
patience_counter = 0
patience = 50

if avg_loss < best_loss:
    best_loss = avg_loss
    patience_counter = 0
    best_model_state = model.state_dict()
else:
    patience_counter += 1

if patience_counter >= patience:
    print(f"Early stopping at epoch {epoch+1}")
    model.load_state_dict(best_model_state)
    break
```

**What this does:**
- Monitors if model is still improving
- Stops training if no improvement for 50 epochs
- Restores best weights

**Why?**
- Prevents overfitting
- Saves time (don't train unnecessarily)

**How it works:**
1. Track best loss seen so far
2. If loss improves → reset counter, save weights
3. If loss doesn't improve → increment counter
4. If counter reaches 50 → stop and restore best weights

---

## 📊 Section 11: Model Comparison

### Step 11.1: Create Comparison Table
```python
comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Neural Network'],
    'Test RMSE': [test_rmse_lr, test_rmse_rf, test_rmse_gb, test_rmse_nn],
    'Test R²': [test_r2_lr, test_r2_rf, test_r2_gb, test_r2_nn]
})

print(comparison_df.sort_values('Test RMSE'))
```

**What this does:**
- Creates a DataFrame comparing all models
- Sorts by RMSE (lower is better)

**Output:**
```
              Model  Test RMSE   Test R²
Gradient Boosting      17,400    0.9934
Random Forest          19,239    0.9920
Linear Regression      42,161    0.9614
Neural Network         64,024    0.9111
```

---

### Step 11.2: Visualization - Bar Charts
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RMSE comparison
axes[0].bar(models, rmse_values, color=['#5B7C99', '#7FA99B', '#C1666B', '#D4A574'])
axes[0].set_ylabel('Test RMSE (€)')
axes[0].set_title('Model Comparison: RMSE')

# R² comparison
axes[1].bar(models, r2_values, color=['#5B7C99', '#7FA99B', '#C1666B', '#D4A574'])
axes[1].set_ylabel('Test R²')
axes[1].set_title('Model Comparison: R²')
```

**What this does:**
- Creates 2 side-by-side bar charts
- Left: RMSE comparison (lower bars = better)
- Right: R² comparison (taller bars = better)

**Colors:**
- Each model gets unique color for easy identification
- Consistent across all visualizations

---

### Step 11.3: Actual vs Predicted Scatter Plot
```python
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_test_pred_gb, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (€)')
plt.ylabel('Predicted Price (€)')
plt.legend()
```

**What this does:**

**Line 2:** Scatter plot
- x-axis = actual prices
- y-axis = predicted prices
- Each dot = one property

**Line 3-4:** Add diagonal reference line
- Represents perfect predictions (actual = predicted)

**How to interpret:**
- Points close to red line → good predictions
- Points above line → over-predicted (too high)
- Points below line → under-predicted (too low)
- Tight cluster → consistent accuracy

---

## 🔍 Section 12: Error Analysis

### Step 12.1: Calculate Residuals
```python
residuals = y_test - y_test_pred_gb
abs_residuals = np.abs(residuals)
```

**What this does:**
- `residuals` = actual - predicted (can be positive or negative)
- `abs_residuals` = absolute value (all positive)

**Example:**
- Actual: €800k, Predicted: €820k
- Residual: -€20k (over-predicted)
- Absolute residual: €20k

---

### Step 12.2: Identify Worst Predictions
```python
worst_indices = abs_residuals.nlargest(10).index

for idx in worst_indices:
    actual = y_test.iloc[idx]
    predicted = y_test_pred_gb[idx]
    error = residuals.iloc[idx]
    print(f"Actual: €{actual:,.0f}, Predicted: €{predicted:,.0f}, Error: €{error:,.0f}")
```

**What this does:**
- Finds 10 properties with largest prediction errors
- Prints details for each

**Why?**
- Understand where model struggles
- Identify patterns in difficult predictions

---

### Step 12.3: Residual Distribution Plot
```python
plt.hist(residuals, bins=30, color='#5B7C99', alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residual (€)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
```

**What this does:**
- Histogram of errors
- Red line at x=0 (perfect prediction)

**What to look for:**
- Centered at 0? (no bias)
- Symmetric? (no systematic over/under-prediction)
- Normal shape? (validates model assumptions)

---

### Step 12.4: Residuals vs Predicted Values
```python
plt.scatter(y_test_pred_gb, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Price (€)')
plt.ylabel('Residual (€)')
```

**What this does:**
- Plots prediction vs error
- Checks for patterns

**What to look for:**
- Random scatter → good (no pattern)
- Funnel shape → heteroscedasticity (variance increases)
- Curve → non-linear relationship missed

---

## 💾 Section 13: Save Results

### Step 13.1: Save Best Model
```python
import pickle

with open('gradient_boosting_model.pkl', 'wb') as f:
    pickle.dump(gb_no_tax, f)
```

**What this does:**
- Saves trained model to disk
- Can load later without retraining

**Usage:**
```python
# Load model later
with open('gradient_boosting_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

---

### Step 13.2: Save Predictions
```python
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_test_pred_gb,
    'Error': residuals
})

results_df.to_csv('predictions.csv', index=False)
```

**What this does:**
- Creates DataFrame with actual, predicted, and errors
- Saves to CSV file

**Why?**
- Document results
- Share with stakeholders
- Further analysis in Excel

---

## 🎯 Key Code Concepts Summary

### Data Flow Through the Pipeline

```
1. Load Data (CSV → DataFrame)
        ↓
2. Clean Data (remove columns, rename)
        ↓
3. Explore Data (plots, statistics)
        ↓
4. Transform Data (log, scale)
        ↓
5. Split Data (train/test)
        ↓
6. Train Models (fit on training data)
        ↓
7. Predict (apply to test data)
        ↓
8. Evaluate (calculate metrics)
        ↓
9. Compare (which model is best?)
        ↓
10. Analyze Errors (where does it fail?)
```

---

## 🔑 Important Code Patterns

### Pattern 1: Always fit on train, transform both
```python
# ✅ Correct
scaler.fit(X_train)  # Learn from training only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ Wrong (data leakage)
scaler.fit(X_test)  # Never fit on test!
```

---

### Pattern 2: Check shapes constantly
```python
print(f"X_train shape: {X_train.shape}")  # (1600, 7)
print(f"X_test shape: {X_test.shape}")    # (400, 7)
```

**Why?** Catch errors early (dimension mismatch, missing data).

---

### Pattern 3: Visualize everything
```python
# Before modeling → understand data
df['price'].hist()

# After modeling → understand predictions
plt.scatter(y_test, y_pred)

# After evaluation → understand errors
plt.hist(residuals)
```

---

## 📝 Common Pitfalls to Avoid

1. **Fitting scaler on test data** → Data leakage
2. **Forgetting to back-transform** → Predictions in wrong scale
3. **Not setting random_state** → Results not reproducible
4. **Using same y_train for different X splits** → Mismatch
5. **Forgetting to drop target from features** → Perfect correlation

---

## 🎓 Next Steps for Understanding

**To deeply understand each section:**

1. **Change parameters** and see what happens
   - Try `n_estimators=50` vs `n_estimators=200`
   - Change `learning_rate=0.01` vs `learning_rate=0.5`

2. **Break the code intentionally**
   - Remove scaling → see how Linear Regression fails
   - Skip log transform → see skewed residuals

3. **Add print statements**
   - Print shapes after each step
   - Print first few predictions

4. **Modify visualizations**
   - Try different colors
   - Add more subplots
   - Change bin sizes

**Remember:** The best way to learn is to experiment!

---

This walkthrough explains every major code block in your notebook. Each section builds on the previous one, creating a complete machine learning pipeline from raw data to final model selection.
