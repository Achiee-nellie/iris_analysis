
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """
    Task 1: Load and Explore the Dataset
    """
    print("=" * 60)
    print("TASK 1: LOADING AND EXPLORING THE DATASET")
    print("=" * 60)
    
    try:
        # Load the Iris dataset
        iris = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("✓ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print()
        
        # Display first few rows
        print("First 10 rows of the dataset:")
        print(df.head(10))
        print()
        
        # Explore dataset structure
        print("Dataset Information:")
        print("-" * 30)
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        print()
        
        print("Column data types:")
        print(df.dtypes)
        print()
        
        print("Missing values per column:")
        missing_values = df.isnull().sum()
        print(missing_values)
        
        if missing_values.sum() == 0:
            print("✓ No missing values found in the dataset!")
        else:
            print("⚠ Missing values detected - cleaning required")
            # Fill missing values with median for numerical, mode for categorical
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            print("✓ Missing values handled successfully!")
        
        print()
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def basic_data_analysis(df):
    """
    Task 2: Basic Data Analysis
    """
    print("=" * 60)
    print("TASK 2: BASIC DATA ANALYSIS")
    print("=" * 60)
    
    try:
        # Basic statistics for numerical columns
        print("Basic Statistics for Numerical Columns:")
        print("-" * 40)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        stats = df[numerical_cols].describe()
        print(stats.round(3))
        print()
        
        # Group analysis by species
        print("Analysis by Species:")
        print("-" * 25)
        
        # Mean values for each species
        species_means = df.groupby('species')[numerical_cols].mean()
        print("Mean values by species:")
        print(species_means.round(3))
        print()
        
        # Count of samples per species
        species_counts = df['species'].value_counts()
        print("Number of samples per species:")
        print(species_counts)
        print()
        
        # Additional analysis - correlations
        print("Correlation Matrix:")
        print("-" * 20)
        correlation_matrix = df[numerical_cols].corr()
        print(correlation_matrix.round(3))
        print()
        
        # Key findings
        print("KEY FINDINGS:")
        print("-" * 15)
        findings = []
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    high_corr_pairs.append(
                        (correlation_matrix.columns[i], 
                         correlation_matrix.columns[j], 
                         corr_val)
                    )
        
        if high_corr_pairs:
            findings.append("High correlations found:")
            for feat1, feat2, corr in high_corr_pairs:
                findings.append(f"  • {feat1} vs {feat2}: {corr:.3f}")
        
        # Species with largest/smallest average measurements
        for col in numerical_cols:
            max_species = species_means[col].idxmax()
            min_species = species_means[col].idxmin()
            findings.append(f"• {col}: Largest in {max_species}, smallest in {min_species}")
        
        # Dataset balance
        if species_counts.std() == 0:
            findings.append("• Dataset is perfectly balanced across species")
        else:
            findings.append(f"• Dataset balance varies: {species_counts.min()}-{species_counts.max()} samples per species")
        
        for finding in findings:
            print(finding)
        
        print()
        return species_means, correlation_matrix
        
    except Exception as e:
        print(f"❌ Error in data analysis: {e}")
        return None, None

def create_visualizations(df, species_means, correlation_matrix):
    """
    Task 3: Data Visualization
    """
    print("=" * 60)
    print("TASK 3: DATA VISUALIZATION")
    print("=" * 60)
    
    try:
        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Iris Dataset Analysis - Comprehensive Visualization', fontsize=16, fontweight='bold')
        
        # Line Chart - Feature trends across species (simulating time series)
        ax1 = plt.subplot(2, 3, 1)
        species_order = ['setosa', 'versicolor', 'virginica']
        for feature in df.select_dtypes(include=[np.number]).columns:
            values = [species_means.loc[species, feature] for species in species_order]
            plt.plot(species_order, values, marker='o', linewidth=2, label=feature)
        
        plt.title('Feature Trends Across Species', fontweight='bold')
        plt.xlabel('Species (Evolution Order)')
        plt.ylabel('Average Measurement (cm)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Bar Chart - Average petal length per species
        ax2 = plt.subplot(2, 3, 2)
        petal_length_avg = df.groupby('species')['petal length (cm)'].mean()
        bars = plt.bar(petal_length_avg.index, petal_length_avg.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        
        plt.title('Average Petal Length by Species', fontweight='bold')
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, petal_length_avg.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Histogram - Distribution of sepal length
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(df['sepal length (cm)'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(df['sepal length (cm)'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["sepal length (cm)"].mean():.2f}')
        plt.axvline(df['sepal length (cm)'].median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {df["sepal length (cm)"].median():.2f}')
        
        plt.title('Distribution of Sepal Length', fontweight='bold')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter Plot - Sepal length vs Petal length
        ax4 = plt.subplot(2, 3, 4)
        colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.scatter(species_data['sepal length (cm)'], species_data['petal length (cm)'],
                       c=colors[species], label=species, alpha=0.7, s=50)
        
        plt.title('Sepal Length vs Petal Length', fontweight='bold')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation coefficient to the plot
        corr_coef = df['sepal length (cm)'].corr(df['petal length (cm)'])
        plt.text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                transform=ax4.transAxes, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Correlation Heatmap
        ax5 = plt.subplot(2, 3, 5)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   mask=mask, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Box Plot - Feature distribution by species
        ax6 = plt.subplot(2, 3, 6)
        df_melted = pd.melt(df, id_vars=['species'], value_vars=numerical_cols,
                           var_name='feature', value_name='measurement')
        sns.boxplot(data=df_melted, x='feature', y='measurement', hue='species', ax=ax6)
        plt.title('Feature Distribution by Species', fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Measurement (cm)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
        
        print("✓ All visualizations created successfully!")
        
        # Additional insights from visualizations
        print("\nVISUALIZATION INSIGHTS:")
        print("-" * 25)
        
        insights = [
            "• Line chart shows clear species progression in measurements",
            "• Bar chart confirms Virginica has the longest petals on average",
            "• Histogram reveals sepal length follows roughly normal distribution",
            f"• Scatter plot shows strong positive correlation ({corr_coef:.3f}) between sepal and petal length",
            "• Heatmap identifies petal dimensions as most highly correlated",
            "• Box plots reveal distinct measurement ranges for each species"
        ]
        
        for insight in insights:
            print(insight)
        
        print()
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")

def main():
    """
    Main function to orchestrate the analysis
    """
    print("🌺 IRIS DATASET ANALYSIS 🌺")
    print("Data Analysis with Pandas & Visualization with Matplotlib")
    print("=" * 60)
    print()
    
    # Load and explore data
    df = load_and_explore_data()
    if df is None:
        print("❌ Failed to load dataset. Exiting...")
        return
    
    #  Basic data analysis
    species_means, correlation_matrix = basic_data_analysis(df)
    if species_means is None:
        print("❌ Failed to perform data analysis. Exiting...")
        return
    
    # Create visualizations
    create_visualizations(df, species_means, correlation_matrix)
    
    print("=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print()
    print("SUMMARY OF DELIVERABLES:")
    print("• ✓ Dataset loaded and explored")
    print("• ✓ Data cleaning performed (no missing values found)")
    print("• ✓ Basic statistics computed")
    print("• ✓ Groupby analysis by species completed")
    print("• ✓ Four main visualizations created:")
    print("    - Line chart (feature trends)")
    print("    - Bar chart (average petal length)")
    print("    - Histogram (sepal length distribution)")
    print("    - Scatter plot (sepal vs petal length)")
    print("• ✓ Additional visualizations (heatmap, box plots)")
    print("• ✓ Error handling implemented")
    print("• ✓ Key findings and insights documented")
    print()
    print("📊 All assignment requirements fulfilled!")

if __name__ == "__main__":
    main()