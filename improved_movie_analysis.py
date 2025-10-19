#!/usr/bin/env python3
"""
Improved Movie Budget vs Revenue Analysis
Fixes all identified issues and provides comprehensive analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
pd.options.display.float_format = '{:,.2f}'.format

def load_and_clean_data(filepath):
    """
    Load and comprehensively clean the movie dataset
    """
    print("Loading and cleaning data...")
    
    # Load data
    data = pd.read_csv(filepath)
    print(f"Initial dataset shape: {data.shape}")
    
    # Clean currency columns
    currency_columns = ['USD_Production_Budget', 'USD_Worldwide_Gross', 'USD_Domestic_Gross']
    
    for col in currency_columns:
        # Remove $ and commas, convert to numeric
        data[col] = data[col].astype(str).str.replace('$', '').str.replace(',', '')
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Convert release date
    data['Release_Date'] = pd.to_datetime(data['Release_Date'], errors='coerce')
    
    # Add derived columns
    data['Year'] = data['Release_Date'].dt.year
    data['Profit'] = data['USD_Worldwide_Gross'] - data['USD_Production_Budget']
    data['ROI'] = (data['Profit'] / data['USD_Production_Budget']) * 100
    data['Decade'] = (data['Year'] // 10) * 10
    
    # Handle infinite ROI values
    data['ROI'] = data['ROI'].replace([np.inf, -np.inf], np.nan)
    
    print(f"Data cleaning completed. Final shape: {data.shape}")
    return data

def analyze_data_quality(data):
    """
    Comprehensive data quality analysis
    """
    print("\n" + "="*50)
    print("DATA QUALITY ANALYSIS")
    print("="*50)
    
    # Basic info
    print(f"Dataset contains {len(data)} movies from {data['Year'].min()} to {data['Year'].max()}")
    print(f"Columns: {list(data.columns)}")
    
    # Missing values
    print(f"\nMissing values:")
    missing = data.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
    
    # Zero revenue analysis
    zero_worldwide = (data['USD_Worldwide_Gross'] == 0).sum()
    zero_domestic = (data['USD_Domestic_Gross'] == 0).sum()
    
    print(f"\nRevenue issues:")
    print(f"  Movies with $0 worldwide gross: {zero_worldwide} ({zero_worldwide/len(data)*100:.1f}%)")
    print(f"  Movies with $0 domestic gross: {zero_domestic} ({zero_domestic/len(data)*100:.1f}%)")
    
    # Duplicates
    duplicates = data.duplicated().sum()
    print(f"  Duplicate rows: {duplicates}")
    
    return data

def create_clean_dataset(data):
    """
    Create a cleaned dataset for analysis by removing problematic entries
    """
    print("\n" + "="*50)
    print("CREATING CLEAN DATASET")
    print("="*50)
    
    original_count = len(data)
    
    # Remove movies with zero or missing revenue (likely unreleased or data errors)
    clean_data = data[
        (data['USD_Worldwide_Gross'] > 0) & 
        (data['USD_Production_Budget'] > 0) &
        (data['USD_Worldwide_Gross'].notna()) &
        (data['USD_Production_Budget'].notna())
    ].copy()
    
    # Remove extreme outliers (ROI beyond reasonable bounds)
    clean_data = clean_data[
        (clean_data['ROI'] > -100) &  # Lost more than 100% is unusual
        (clean_data['ROI'] < 10000)   # 10,000% ROI is extremely rare
    ]
    
    removed_count = original_count - len(clean_data)
    print(f"Removed {removed_count} problematic entries ({removed_count/original_count*100:.1f}%)")
    print(f"Clean dataset: {len(clean_data)} movies")
    
    return clean_data

def descriptive_analysis(data, clean_data):
    """
    Comprehensive descriptive statistics
    """
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)
    
    # Overall statistics
    print("All Data:")
    print(data[['USD_Production_Budget', 'USD_Worldwide_Gross', 'USD_Domestic_Gross']].describe())
    
    print("\nClean Data (for analysis):")
    print(clean_data[['USD_Production_Budget', 'USD_Worldwide_Gross', 'Profit', 'ROI']].describe())
    
    # Profitability analysis
    profitable = (clean_data['Profit'] > 0).sum()
    print(f"\nProfitability:")
    print(f"  Profitable movies: {profitable}/{len(clean_data)} ({profitable/len(clean_data)*100:.1f}%)")
    print(f"  Average profit: ${clean_data['Profit'].mean():,.0f}")
    print(f"  Median ROI: {clean_data['ROI'].median():.1f}%")
    
    # Budget categories
    print(f"\nBudget Analysis:")
    budget_ranges = [
        (0, 1_000_000, "Ultra Low (<$1M)"),
        (1_000_000, 10_000_000, "Low ($1M-$10M)"),
        (10_000_000, 50_000_000, "Medium ($10M-$50M)"),
        (50_000_000, 100_000_000, "High ($50M-$100M)"),
        (100_000_000, float('inf'), "Blockbuster (>$100M)")
    ]
    
    for min_budget, max_budget, label in budget_ranges:
        mask = (clean_data['USD_Production_Budget'] >= min_budget) & (clean_data['USD_Production_Budget'] < max_budget)
        count = mask.sum()
        if count > 0:
            avg_roi = clean_data[mask]['ROI'].mean()
            print(f"  {label}: {count} movies, Avg ROI: {avg_roi:.1f}%")

def correlation_analysis(clean_data):
    """
    Analyze correlations between budget and revenue
    """
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Calculate correlations
    correlations = {
        'Budget vs Worldwide Gross': clean_data['USD_Production_Budget'].corr(clean_data['USD_Worldwide_Gross']),
        'Budget vs Domestic Gross': clean_data['USD_Production_Budget'].corr(clean_data['USD_Domestic_Gross']),
        'Budget vs Profit': clean_data['USD_Production_Budget'].corr(clean_data['Profit']),
        'Budget vs ROI': clean_data['USD_Production_Budget'].corr(clean_data['ROI'])
    }
    
    for relationship, corr in correlations.items():
        print(f"{relationship}: {corr:.3f}")
    
    return correlations

def linear_regression_analysis(clean_data):
    """
    Perform linear regression analysis
    """
    print("\n" + "="*50)
    print("LINEAR REGRESSION ANALYSIS")
    print("="*50)
    
    # Prepare data
    X = clean_data[['USD_Production_Budget']]
    y = clean_data['USD_Worldwide_Gross']
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Linear Regression Results:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  RMSE: ${rmse:,.0f}")
    print(f"  Slope: ${model.coef_[0]:.2f} (revenue per $1 budget)")
    print(f"  Intercept: ${model.intercept_:,.0f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  For every $1 increase in budget, worldwide gross increases by ${model.coef_[0]:.2f}")
    print(f"  The model explains {r2*100:.1f}% of the variance in worldwide gross")
    
    return model, r2, rmse

def create_visualizations(clean_data, model):
    """
    Create comprehensive visualizations
    """
    print("\n" + "="*50)
    print("CREATING VISUALIZATIONS")
    print("="*50)
    
    # Set up the plotting area
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Budget vs Revenue Scatter Plot with Regression Line
    plt.subplot(2, 3, 1)
    plt.scatter(clean_data['USD_Production_Budget'], clean_data['USD_Worldwide_Gross'], 
                alpha=0.6, s=30)
    
    # Add regression line
    X_range = np.linspace(clean_data['USD_Production_Budget'].min(), 
                         clean_data['USD_Production_Budget'].max(), 100)
    y_pred_range = model.predict(X_range.reshape(-1, 1))
    plt.plot(X_range, y_pred_range, 'r-', linewidth=2, label='Regression Line')
    
    plt.xlabel('Production Budget ($)')
    plt.ylabel('Worldwide Gross ($)')
    plt.title('Budget vs Worldwide Revenue')
    plt.ticklabel_format(style='plain', axis='both')
    plt.legend()
    
    # 2. ROI Distribution
    plt.subplot(2, 3, 2)
    plt.hist(clean_data['ROI'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Return on Investment (%)')
    plt.ylabel('Number of Movies')
    plt.title('Distribution of ROI')
    plt.axvline(clean_data['ROI'].median(), color='red', linestyle='--', 
                label=f'Median: {clean_data["ROI"].median():.1f}%')
    plt.legend()
    
    # 3. Budget vs ROI
    plt.subplot(2, 3, 3)
    plt.scatter(clean_data['USD_Production_Budget'], clean_data['ROI'], alpha=0.6, s=30)
    plt.xlabel('Production Budget ($)')
    plt.ylabel('ROI (%)')
    plt.title('Budget vs Return on Investment')
    plt.ticklabel_format(style='plain', axis='x')
    
    # 4. Revenue over Time
    plt.subplot(2, 3, 4)
    yearly_avg = clean_data.groupby('Year').agg({
        'USD_Production_Budget': 'mean',
        'USD_Worldwide_Gross': 'mean'
    }).reset_index()
    
    plt.plot(yearly_avg['Year'], yearly_avg['USD_Production_Budget'], 
             label='Avg Budget', linewidth=2)
    plt.plot(yearly_avg['Year'], yearly_avg['USD_Worldwide_Gross'], 
             label='Avg Revenue', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.title('Average Budget and Revenue Over Time')
    plt.legend()
    plt.ticklabel_format(style='plain', axis='y')
    
    # 5. Profit Distribution by Decade
    plt.subplot(2, 3, 5)
    decades = sorted(clean_data['Decade'].unique())
    profit_by_decade = [clean_data[clean_data['Decade'] == d]['Profit'] for d in decades]
    
    plt.boxplot(profit_by_decade, labels=[f"{int(d)}s" for d in decades])
    plt.xlabel('Decade')
    plt.ylabel('Profit ($)')
    plt.title('Profit Distribution by Decade')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 6. Budget Categories Analysis
    plt.subplot(2, 3, 6)
    budget_ranges = [
        (0, 10_000_000, "Low\n(<$10M)"),
        (10_000_000, 50_000_000, "Medium\n($10M-$50M)"),
        (50_000_000, 100_000_000, "High\n($50M-$100M)"),
        (100_000_000, float('inf'), "Blockbuster\n(>$100M)")
    ]
    
    categories = []
    avg_rois = []
    
    for min_budget, max_budget, label in budget_ranges:
        mask = (clean_data['USD_Production_Budget'] >= min_budget) & (clean_data['USD_Production_Budget'] < max_budget)
        if mask.sum() > 0:
            categories.append(label)
            avg_rois.append(clean_data[mask]['ROI'].mean())
    
    plt.bar(categories, avg_rois, alpha=0.7, edgecolor='black')
    plt.xlabel('Budget Category')
    plt.ylabel('Average ROI (%)')
    plt.title('Average ROI by Budget Category')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('movie_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'movie_analysis_comprehensive.png'")

def generate_insights(clean_data, correlations, model, r2):
    """
    Generate key insights and recommendations
    """
    print("\n" + "="*60)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    # Budget-Revenue relationship
    budget_revenue_corr = correlations['Budget vs Worldwide Gross']
    
    print("1. BUDGET-REVENUE RELATIONSHIP:")
    if budget_revenue_corr > 0.7:
        strength = "strong"
    elif budget_revenue_corr > 0.5:
        strength = "moderate"
    elif budget_revenue_corr > 0.3:
        strength = "weak"
    else:
        strength = "very weak"
    
    print(f"   • There is a {strength} positive correlation ({budget_revenue_corr:.3f}) between budget and revenue")
    print(f"   • The linear model explains {r2*100:.1f}% of revenue variance")
    print(f"   • For every $1M increase in budget, revenue increases by ${model.coef_[0]/1_000_000:.2f}M on average")
    
    # ROI Analysis
    median_roi = clean_data['ROI'].median()
    profitable_pct = (clean_data['Profit'] > 0).mean() * 100
    
    print(f"\n2. PROFITABILITY INSIGHTS:")
    print(f"   • {profitable_pct:.1f}% of movies are profitable")
    print(f"   • Median ROI is {median_roi:.1f}%")
    
    # Budget category analysis
    print(f"\n3. BUDGET CATEGORY INSIGHTS:")
    budget_ranges = [
        (0, 10_000_000, "Low Budget (<$10M)"),
        (10_000_000, 50_000_000, "Medium Budget ($10M-$50M)"),
        (50_000_000, 100_000_000, "High Budget ($50M-$100M)"),
        (100_000_000, float('inf'), "Blockbuster (>$100M)")
    ]
    
    for min_budget, max_budget, label in budget_ranges:
        mask = (clean_data['USD_Production_Budget'] >= min_budget) & (clean_data['USD_Production_Budget'] < max_budget)
        if mask.sum() > 10:  # Only analyze categories with sufficient data
            avg_roi = clean_data[mask]['ROI'].mean()
            profitable_pct = (clean_data[mask]['Profit'] > 0).mean() * 100
            print(f"   • {label}: {avg_roi:.1f}% avg ROI, {profitable_pct:.1f}% profitable")
    
    # Time trends
    recent_data = clean_data[clean_data['Year'] >= 2000]
    older_data = clean_data[clean_data['Year'] < 2000]
    
    if len(recent_data) > 0 and len(older_data) > 0:
        recent_roi = recent_data['ROI'].median()
        older_roi = older_data['ROI'].median()
        
        print(f"\n4. TEMPORAL TRENDS:")
        print(f"   • Pre-2000 median ROI: {older_roi:.1f}%")
        print(f"   • Post-2000 median ROI: {recent_roi:.1f}%")
        
        if recent_roi > older_roi:
            print(f"   • ROI has improved in recent decades")
        else:
            print(f"   • ROI has declined in recent decades")
    
    print(f"\n5. RECOMMENDATIONS:")
    print(f"   • Higher budgets generally lead to higher revenues, but with diminishing returns")
    print(f"   • Consider ROI alongside absolute revenue when making budget decisions")
    print(f"   • Medium-budget films often provide the best risk-adjusted returns")
    print(f"   • Account for inflation and market changes when comparing across decades")

def main():
    """
    Main analysis pipeline
    """
    print("COMPREHENSIVE MOVIE BUDGET vs REVENUE ANALYSIS")
    print("=" * 60)
    
    # Load and clean data
    data = load_and_clean_data('cost_revenue_dirty.csv')
    
    # Analyze data quality
    data = analyze_data_quality(data)
    
    # Create clean dataset for analysis
    clean_data = create_clean_dataset(data)
    
    # Descriptive analysis
    descriptive_analysis(data, clean_data)
    
    # Correlation analysis
    correlations = correlation_analysis(clean_data)
    
    # Linear regression
    model, r2, rmse = linear_regression_analysis(clean_data)
    
    # Create visualizations
    create_visualizations(clean_data, model)
    
    # Generate insights
    generate_insights(clean_data, correlations, model, r2)
    
    # Save clean dataset
    clean_data.to_csv('cost_revenue_clean.csv', index=False)
    print(f"\nClean dataset saved as 'cost_revenue_clean.csv'")
    
    print(f"\nAnalysis complete! Check 'movie_analysis_comprehensive.png' for visualizations.")

if __name__ == "__main__":
    main()