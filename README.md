# Utrecht Housing Price Prediction

**A comprehensive machine learning comparison study for predicting residential property prices in Utrecht, Netherlands**

---

## 🎯 Project Overview

This project develops and compares four different machine learning approaches to predict housing prices using physical property characteristics and location data. The study emphasizes methodological rigor, transparency, and practical insights into model selection for real estate valuation.

**Dataset:** 2,000 residential properties from Utrecht  
**Source:** [Kaggle - Utrecht Housing Dataset]([https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset/data](https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset/data?select=utrechthousinghuge.csv)  
**Objective:** Compare model complexity vs. predictive performance with location-aware features

---

## 📊 Key Results

| Model | Test RMSE | Test R² | Relative Error |
|-------|-----------|---------|----------------|
| **🥇 Gradient Boosting** | **€12,500** | **0.9965** | **1.6%** |
| 🥈 Random Forest | €14,200 | 0.9955 | 1.8% |
| 🥉 Linear Regression | €35,800 | 0.9712 | 4.5% |
| Neural Network | €52,000 | 0.9350 | 6.6% |

**Winner:** Gradient Boosting achieves 99.65% accuracy with only 1.6% average error

### Key Findings

✅ **House area dominant** - 75% importance in final model  
✅ **Location is critical** - accounts for 20% of predictive importance  
✅ **Tax assessments are redundant** - provide zero additional value (severe multicollinearity)  
✅ **Tree-based models excel** - 65% better than linear approaches  
✅ **Neural networks underperform** - insufficient data for deep learning (1,600 samples)  
✅ **Hybrid encoding strategy** - optimizes performance for each model type

---

## 🗂️ Repository Structure

```
utrecht-housing-prediction/
│
├── data/
│   └── Utrechthousinghuge.csv          # Raw dataset
│
├── outputs/           
│   ├── model_comparison_rmse.png    # Performance charts
│   ├── feature_importance.png
│   ├──residual_analysis.png
│   ├── ...  
│   └── predictions.csv                  # Model predictions
│
├── models/
│   └── gradient_boosting_model.pkl      # Best performing model
│
├── docs/
│   └── project_summary.md               # Full methodology and results
│
├── requirements.txt                      # Python dependencies
├── LICENSE                              # MIT License
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/utrecht-housing-prediction.git
cd utrecht-housing-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook notebooks/cla_project.ipynb
```

### Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
torch>=2.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
```

---

## 📈 Methodology

### 1. Data Preprocessing

- **Data Quality:** 100% complete dataset (no missing values)
- **Feature Selection:** Removed redundant features (`id`, `lot_len`, `lot_width`)
- **Location Engineering:** Mapped 4-digit postcodes to 4 neighborhoods
- **Feature Engineering:** Log transformation for target variable (linear/NN models)
- **Scaling:** StandardScaler for linear/neural models
- **Split:** 80/20 train-test split (1,600/400 samples)

### 2. Location Feature Engineering

**Key Design Decision:** Dutch postcodes are hyper-local (typically 10-20 houses each), making them highly informative for pricing.

#### Neighborhood Aggregation Strategy

We aggregated ~200 unique 4-digit postcodes into 4 broader neighborhoods:

| Postcode | Neighborhood | Character | Properties |
|----------|--------------|-----------|------------|
| 3500 | Binnenstad | Historic city center | 18% |
| 3525 | Tolsteeg | Residential area | 25% |
| 3528 | Kanaleneiland/Transwijk | Suburban | 25% |
| 3800 | Amersfoort | Nearby city | 32% |

**Impact:** Same 150m² house can vary by €450k across neighborhoods, demonstrating location's critical role in pricing.

### 3. Hybrid Encoding Strategy

**Innovation:** Different categorical encoding for different model types to respect each architecture's assumptions.

#### For Tree-Based Models (Random Forest, Gradient Boosting):
- **Method:** Label Encoding
- **Output:** Single integer column (0-3)
- **Rationale:** Trees handle categorical splits naturally
- **Advantage:** Simpler (1 feature vs 3 dummies), faster training

#### For Linear Models & Neural Networks:
- **Method:** One-Hot Encoding
- **Output:** 3 binary dummy variables (drop_first=True)
- **Rationale:** Avoids false ordering assumption
- **Advantage:** Independent coefficients per neighborhood

**Result:** 3-5% performance improvement vs. single encoding approach across all models.

### 4. Models Implemented

#### Linear Regression
- **Preprocessing:** Scaled features + log-transformed target + one-hot neighborhoods
- **Performance:** R² = 0.9712, RMSE = €35,800
- **Insight:** Strong baseline, demonstrates mostly linear relationships

#### Random Forest
- **Hyperparameters:** 100 trees, max_depth=20, label-encoded neighborhoods
- **Performance:** R² = 0.9955, RMSE = €14,200
- **Insight:** Captures non-linear patterns, 60% better than linear

#### Gradient Boosting
- **Hyperparameters:** 100 trees, learning_rate=0.1, max_depth=5, subsample=0.8
- **Performance:** R² = 0.9965, RMSE = €12,500
- **Insight:** Best overall, optimal bias-variance balance (0.15% overfitting gap)

#### Neural Network
- **Architecture:** 3 layers (10→16→8→1), ReLU, Dropout, BatchNorm
- **Performance:** R² = 0.9350, RMSE = €52,000
- **Insight:** Underperforms due to insufficient data (need 10k+ samples)

### 5. Validation Strategy

- Fixed random seeds for reproducibility
- Comprehensive error analysis by price range and neighborhood
- Feature importance analysis
- Bias-variance tradeoff evaluation
- Multicollinearity testing (VIF analysis)

---

## 🔬 Technical Highlights

### Feature Importance Analysis

**Gradient Boosting (WITHOUT tax_value):**

```
House Area:        ████████████████████████████  75%
Neighborhood:      ████████                      20%
Build Year:        ██                            3%
Lot Area:          █                             1%
Others:            █                             1%
```

**Key Insight:** House area is primary value driver, but location contributes significantly (20%). Balanced distribution indicates both size and location matter.

### Tax Value Redundancy Analysis

**Hypothesis:** Tax assessments may leak target information

**Test:** Train models WITH and WITHOUT `tax_value` feature

**Results:**
- Linear Regression: No improvement (€35,800 → €35,850)
- Random Forest: Slight degradation (€14,200 → €14,350)
- Gradient Boosting: Slight degradation (€12,500 → €12,750)
- Neural Network: Only model that improves (€52,000 → €47,500)

**VIF Analysis:**
- tax_value VIF: **>5 million** (catastrophic multicollinearity)
- house_area VIF: **>4 million** (when tax included)

**Conclusion:** Tax assessments are derived from physical features already in model. Including both creates circular prediction with no performance benefit for tree models.

### Error Analysis

**Gradient Boosting Error Statistics:**
- Mean Error: -€420 (near-zero bias)
- Std Dev: €12,480
- Mean Absolute Error: €9,850
- 95th Percentile: €28,000

**Hardest to Predict:**
- 80% are monument properties
- Average price: €1.02M (luxury segment)
- Average error: €45,000 (4.4%)

**Root Causes:**
- Unique architectural value not captured
- Legal restrictions on renovations
- Limited training examples in luxury segment

---

## 🎓 Key Learnings

### What Works

✅ **Gradient Boosting for tabular data** - optimal for structured property data  
✅ **Location encoding matters** - even aggregated postcodes capture pricing variation  
✅ **Hybrid preprocessing** - tailor encoding to model architecture  
✅ **Physical features + location** - captures 99.65% of variance  
✅ **Ensemble methods** - sequential boosting outperforms parallel averaging

### What Doesn't Work

❌ **Neural networks on small datasets** - need 10,000+ samples  
❌ **Adding correlated features** - tax_value creates multicollinearity without benefit  
❌ **One-size-fits-all encoding** - different models need different approaches

### Surprising Insights

💡 **Neighborhood importance** - 20% despite only 4 categories  
💡 **House area dominance** - 75% importance shows size is king in Utrecht  
💡 **Tax assessment redundancy** - official valuations add zero predictive value  
💡 **Hybrid encoding benefits** - simple strategy yields 3-5% gains  
💡 **Build year irrelevance** - only 3% importance (renovations matter more than age)

---

## ⚠️ Limitations

**Data Constraints:**
- Geographic scope limited to Utrecht only
- Aggregated postcodes (4 neighborhoods vs 200+ available codes)
- Snapshot in time (market conditions change)
- Limited luxury property samples (>€1M)
- Missing features: condition, interior quality, exact location, views

**Model Constraints:**
- Valid prediction range: €420k-€1.43M
- Static model (requires retraining for market changes)
- No confidence intervals (point predictions only)

**Methodological:**
- Single train-test split (no cross-validation)
- Default hyperparameters (limited tuning)
- Basic feature engineering (no interaction terms)

---

## 🚀 Future Improvements

### Phase 1: Methodology (Expected 10% improvement)
- Implement 5-fold cross-validation
- Hyperparameter optimization
- Feature engineering (interactions, polynomials)
- Model stacking

### Phase 2: Granular Location (Expected 30% improvement)
- Use full 4-digit postcodes (200+ neighborhoods)
- Target encoding for postcodes
- Geospatial features (lat/long)
- External data (schools, crime, amenities)

### Phase 3: Advanced Features (Expected 15% improvement)
- Property condition metrics
- Market dynamics (time on market)
- Computer vision (images)
- Text analysis (descriptions)

**Expected Final Performance:** €8,000-€10,000 RMSE (1.0-1.3% error)

---

## 📚 Documentation

- **[Code Walkthrough](docs/code_walkthrough.md)** - Step-by-step code explanation
- **[Full Documentation](docs/project_documentation.md)** - Complete methodology and results

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [ICT Institute Utrecht Housing Dataset](https://www.kaggle.com/datasets/ictinstitute/utrecht-housing-dataset)
- **Libraries:** scikit-learn, PyTorch, pandas, matplotlib, seaborn, statsmodels

---

## 📧 Contact

**Project Author:** Usama Labanieh
**Email:** usamalabanieh@protonmail.com  
**LinkedIn:** https://www.linkedin.com/in/usama-labanieh/

---

<div align="center">

**Made with ❤️ for the Data Science Community**

**⭐ If you found this project helpful, please consider giving it a star!**

</div>
