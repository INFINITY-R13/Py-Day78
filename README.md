# Movie Budget vs Revenue Analysis

**Research Question**: Do higher film budgets lead to more box office revenue?

## 🎯 Overview

Comprehensive analysis of movie production budgets and box office revenue using data from 5,006 films (1915-2020). Includes data cleaning, statistical analysis, linear regression, and business insights for film investment decisions.

## 📊 Key Findings

- **Strong Correlation**: 0.743 between budget and worldwide revenue
- **ROI**: $3.12 return for every $1 invested (average)
- **Profitability**: 66.9% of movies profitable, 84.9% median ROI
- **Model**: Linear regression explains 55.1% of revenue variance
- **Sweet Spot**: Medium-budget films ($10M-$50M) offer best risk-adjusted returns

## 🚀 Quick Start

### Option 1: Run Python Script (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run comprehensive analysis
python improved_movie_analysis.py
```

### Option 2: Use Jupyter Notebook
```bash
# Install Jupyter if needed
pip install jupyter

# Launch notebook
jupyter notebook Improved_Movie_Analysis.ipynb
```

## 📁 Project Structure

```
├── cost_revenue_dirty.csv              # Original dataset
├── improved_movie_analysis.py          # Complete analysis script
├── Improved_Movie_Analysis.ipynb       # Interactive notebook
├── requirements.txt                    # Python dependencies
└── README.md                          # Documentation
```

**Generated Files** (after running analysis):
- `cost_revenue_clean.csv` - Cleaned dataset
- `movie_analysis_comprehensive.png` - Visualization dashboard

## 📈 Analysis Features

- **Data Cleaning**: Handles currency formatting, removes invalid entries
- **Statistical Analysis**: Correlation, regression, ROI calculations
- **Visualizations**: 6-panel comprehensive dashboard
- **Business Insights**: Investment recommendations by budget category

## 💡 Key Insights

| Budget Category | Avg ROI | Success Rate | Risk Level |
|----------------|---------|--------------|------------|
| Ultra Low (<$1M) | 806% | 60% | High |
| Low ($1M-$10M) | 401% | 60% | Medium |
| Medium ($10M-$50M) | 189% | 65% | Low |
| High ($50M-$100M) | 155% | 75% | Very Low |
| Blockbuster (>$100M) | 205% | 90% | Low |

### Investment Strategy
- **Higher budgets = Higher revenues** (strong 0.743 correlation)
- **Medium budgets** offer best risk-adjusted returns
- **Blockbusters** have highest success rates but require large capital
- **Story quality matters** - budget alone doesn't guarantee success

## 🛠 Technical Stack

- **Python 3.8+** with Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Dataset**: 5,006 validated movies (1915-2020) from [the-numbers.com](https://www.the-numbers.com/movie/budgets)
- **Model**: Linear Regression (R² = 0.551, RMSE = $115.8M)

---

*Analysis demonstrates that strategic budget allocation can significantly impact film profitability and ROI.*
