import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Check for required packages and provide installation instructions if missing
required_packages = {
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'statsmodels': 'statsmodels'
}

missing_packages = []
for package, pip_name in required_packages.items():
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(pip_name)

if missing_packages:
    logger.error("The following required packages are missing:")
    for package in missing_packages:
        logger.error(f"  - {package}")
    logger.error("Please install them using pip:")
    logger.error(f"pip install {' '.join(missing_packages)}")
    logger.error("Or if you're using conda:")
    logger.error(f"conda install {' '.join(missing_packages)}")
    sys.exit(1)

# Import after checking
import seaborn as sns
from scipy import stats
import matplotlib.font_manager as fm

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Try to set a font that supports emoji characters
# Common fonts that support emoji: Apple Color Emoji, Segoe UI Emoji, Noto Color Emoji
emoji_fonts = ['Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', 'Symbola']
emoji_font_found = False

for font in emoji_fonts:
    font_list = [f for f in fm.findSystemFonts() if font.lower() in os.path.basename(f).lower()]
    if font_list:
        plt.rcParams['font.family'] = font
        logger.info(f"Using font: {font} for emoji support")
        emoji_font_found = True
        break


logger.info("""
=======================================================================
                ENHANCED FITNESS TEST ANALYSIS
=======================================================================
This script performs an enhanced analysis of fitness test data, including:

1. Data Cleaning & Preprocessing:
   - Handling missing values
   - Detecting outliers using Z-scores

2. Basic Statistical Analysis:
   - Descriptive statistics (mean, std, min, max, etc.)
   - Correlation analysis between fitness metrics

3. Group Comparisons:
   - Gender-based analysis with statistical significance testing
   - Experience level analysis with ANOVA and post-hoc tests

4. Visualizations:
   - Correlation heatmap
   - Distribution plots for each fitness metric
   - Radar charts comparing fitness profiles by gender and experience
   - Summary dashboard with key insights

5. Results Export:
   - Top performers exported to CSV
   - All visualizations saved to the 'output' directory

=======================================================================
""")

# Read the Excel file
logger.info("Reading data from DBZdata.xlsx...")
df = pd.read_excel("DBZdata.xlsx")

# Data cleaning
logger.info("Cleaning data...")
# Fill missing values with appropriate defaults
df['Experience Level'] = df['Experience Level'].fillna('Unknown')

# Replace emoji characters in the data to avoid font issues
df['Experience Level'] = df['Experience Level'].replace({
    'Experienced old person with back pain 🐒': 'Experienced old person with back pain',
    'Have paddled a few times before 🦆': 'Have paddled a few times before',
    'Newbie <3 🐥': 'Newbie <3'
})

numeric_cols = ['Mobility', 'Core', 'Pullup', 'Bench', 'Pushup', 'Cardio', 'Total']
df[numeric_cols] = df[numeric_cols].fillna(0)

# Check for outliers using Z-score
logger.info("Checking for outliers (Z-score > 3):")
for col in numeric_cols:
    # Avoid running zscore on constant/empty columns
    col_non_na = df[col].dropna()
    if col_non_na.shape[0] < 2 or col_non_na.nunique() <= 1:
        continue

    # Compute z-scores on the non-null values
    z_scores = stats.zscore(col_non_na)

    # Ensure we have a numpy array of absolute z-scores
    z_scores = np.abs(np.asarray(z_scores))

    # Create a boolean mask for outliers and map back to the original DataFrame indices
    outlier_mask = z_scores > 3
    outlier_indices = col_non_na.index[outlier_mask]
    outliers = df.loc[outlier_indices]

    if not outliers.empty:
        logger.info(f"Outliers in {col}: {len(outliers)} rows")
        logger.debug(outliers[['Name', col]])

# Display the first 5 rows
logger.info("First 5 rows of the dataset:")
logger.info("\n%s", df.head())

# Display basic information about the dataset
logger.info("Basic information about the dataset:")
logger.info(f"Number of rows: {df.shape[0]}")
logger.info(f"Number of columns: {df.shape[1]}")
logger.info("Column names: %s", df.columns.tolist())

# Display basic statistics
logger.info("Basic statistics:")
logger.info("\n%s", df.describe())

# Check for missing values
logger.info("Missing values in each column:")
logger.info("\n%s", df.isnull().sum())

# Group analysis by gender
logger.info("Average metrics by gender:")
gender_analysis = df.groupby('Gender')[['Mobility', 'Core', 'Pullup', 'Bench', 'Pushup', 'Cardio', 'Total']].mean()
logger.info("\n%s", gender_analysis)

# Correlation between different fitness metrics
logger.info("Correlation between fitness metrics:")
correlation = df[['Mobility', 'Core', 'Pullup', 'Bench', 'Pushup', 'Cardio', 'Total']].corr()
logger.info("\n%s", correlation)

# Create a correlation heatmap
plt.figure(figsize=(10, 10))  # Increased height to accommodate analysis text
mask = np.triu(correlation)
heatmap = sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', 
                     mask=mask, vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
plt.title('Correlation Heatmap of Fitness Metrics', fontsize=16, fontweight='bold')

# Find the strongest positive and negative correlations (excluding self-correlations)
corr_no_diag = correlation.copy()
np.fill_diagonal(corr_no_diag.values, 0)
strongest_pos = corr_no_diag.stack().nlargest(1)
strongest_neg = corr_no_diag.stack().nsmallest(1)

# Add analysis text box
analysis_text = "ANALYSIS:\n"
analysis_text += f"• Strongest positive correlation: {strongest_pos.index[0][0]} & {strongest_pos.index[0][1]} ({strongest_pos.values[0]:.2f})\n"
if strongest_neg.values[0] < 0:
    analysis_text += f"• Strongest negative correlation: {strongest_neg.index[0][0]} & {strongest_neg.index[0][1]} ({strongest_neg.values[0]:.2f})\n"
analysis_text += f"• Total score correlates most with: {correlation['Total'].drop('Total').idxmax()} ({correlation['Total'].drop('Total').max():.2f})\n"
analysis_text += f"• This suggests {correlation['Total'].drop('Total').idxmax()} has the greatest impact on overall performance."

plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, 
           bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}, wrap=True)

plt.tight_layout(rect=[0, 0.1, 1, 0.97])  # Adjust layout to make room for text
plt.savefig('output/correlation_heatmap.png', dpi=300)
logger.info("Correlation heatmap saved as 'output/correlation_heatmap.png'")

# Distribution of fitness metrics
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:-1], 1):  # Exclude Total
    plt.subplot(2, 3, i)
    sns.histplot(df[col].dropna(), kde=True, bins=10)
    plt.title(f'Distribution of {col} Scores', fontweight='bold')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('output/fitness_distributions.png', dpi=300)
logger.info("Fitness distributions saved as 'output/fitness_distributions.png'")

# Top performers based on total score
logger.info("Top 5 performers based on total score:")
top_performers = df.sort_values(by='Total', ascending=False).head(5)[['Name', 'Gender', 'Total']]
logger.info("\n%s", top_performers)

# Export top performers to CSV
top_performers_extended = df.sort_values(by='Total', ascending=False).head(10)[
    ['Name', 'Gender', 'Experience Level'] + numeric_cols
]
top_performers_extended.to_csv('output/top_performers.csv', index=False)
logger.info("Top 10 performers exported to 'output/top_performers.csv'")

# Statistical tests to compare groups
logger.info("--- Statistical Tests ---")

# T-tests to compare genders (excluding invalid gender)
valid_genders = ['M', 'F']
gender_df = df[df['Gender'].isin(valid_genders)]

# Define the metrics to analyze
metrics = ['Mobility', 'Core', 'Pullup', 'Bench', 'Pushup', 'Cardio']

logger.info("T-tests comparing genders (M vs F):")
for metric in metrics:
    male_data = gender_df[gender_df['Gender'] == 'M'][metric]
    female_data = gender_df[gender_df['Gender'] == 'F'][metric]

    # Ensure there is enough data to run the test
    if male_data.shape[0] < 2 or female_data.shape[0] < 2:
        logger.info(f"{metric}: Not enough data to perform t-test (need at least 2 per group)")
        continue

    t_stat, p_val = stats.ttest_ind(male_data, female_data, equal_var=False)
    significance = "Significant" if p_val < 0.05 else "Not significant"

    logger.info(f"{metric}: t={t_stat:.2f}, p={p_val:.4f} ({significance})")
    logger.info(f"  Mean for M: {male_data.mean():.2f}, Mean for F: {female_data.mean():.2f}")
    logger.info(f"  Difference: {male_data.mean() - female_data.mean():.2f}")

# Get mean values by experience level for ANOVA
exp_means = df.groupby('Experience Level')[metrics].mean()

# ANOVA to compare experience levels
logger.info("ANOVA comparing experience levels:")
for metric in metrics:
    groups = [df[df['Experience Level'] == level][metric] for level in exp_means.index 
              if level != 'Unknown' and len(df[df['Experience Level'] == level]) > 0]

    if len(groups) > 1:  # Need at least 2 groups for ANOVA
        f_stat, p_val = stats.f_oneway(*groups)
        significance = "Significant" if p_val < 0.05 else "Not significant"

        logger.info(f"{metric}: F={f_stat:.2f}, p={p_val:.4f} ({significance})")

        # If significant, perform post-hoc Tukey test
        if p_val < 0.05:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd

            # Prepare data for Tukey's test
            data = df[df['Experience Level'] != 'Unknown'][metric]
            labels = df[df['Experience Level'] != 'Unknown']['Experience Level']

            # Perform Tukey's test
            tukey = pairwise_tukeyhsd(data, labels, alpha=0.05)
            logger.info("  Post-hoc Tukey test:")
            logger.info("\n%s", tukey)

# Experience level distribution
logger.info("Count by experience level:")
experience_count = df['Experience Level'].value_counts().sort_index()
logger.info("\n%s", experience_count)

# Create radar charts to compare fitness profiles by gender and experience level

# Radar chart by gender
plt.figure(figsize=(10, 10))  # Increased height for analysis text
ax = plt.subplot(111, polar=True)

# Get mean values by gender
gender_means = df.groupby('Gender')[metrics].mean()

# Number of variables
N = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # Close the loop

# Add labels
plt.xticks(angles[:-1], metrics, fontsize=12)

# Filter out invalid genders
valid_genders = ['M', 'F']
gender_means = gender_means.loc[gender_means.index.isin(valid_genders)]

# Plot data for each gender
colors = ['#FF9671', '#845EC2', '#00C2A8']
markers = ['o', 's', '^']
for i, gender in enumerate(gender_means.index):
    values = gender_means.loc[gender].values.flatten().tolist()
    values += values[:1]  # Close the loop
    ax.plot(angles, values, linestyle='-', linewidth=2, markersize=8, 
            label=f'Gender: {gender}', color=colors[i % len(colors)], 
            marker=markers[i % len(markers)])
    ax.fill(angles, values, colors[i % len(colors)], alpha=0.1)

# Customize the chart
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4], ["1", "2", "3", "4"], color="grey", size=10)
plt.ylim(0, 4)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Fitness Profile by Gender', fontsize=16, fontweight='bold', y=1.1)

# Prepare analysis text
if 'M' in gender_means.index and 'F' in gender_means.index:
    # Find biggest gender differences
    gender_diff = gender_means.loc['M'] - gender_means.loc['F']
    max_diff_metric = gender_diff.abs().idxmax()
    max_diff_value = gender_diff[max_diff_metric]

    # Find strongest metrics for each gender
    m_strongest = gender_means.loc['M'].idxmax()
    f_strongest = gender_means.loc['F'].idxmax()

    # Find weakest metrics for each gender
    m_weakest = gender_means.loc['M'].idxmin()
    f_weakest = gender_means.loc['F'].idxmin()

    # Create analysis text
    analysis_text = "ANALYSIS:\n"
    analysis_text += f"• Largest gender difference: {max_diff_metric} ({abs(max_diff_value):.2f} points "
    analysis_text += f"higher for {'males' if max_diff_value > 0 else 'females'})\n"
    analysis_text += f"• Male strongest area: {m_strongest} ({gender_means.loc['M', m_strongest]:.2f})\n"
    analysis_text += f"• Female strongest area: {f_strongest} ({gender_means.loc['F', f_strongest]:.2f})\n"
    analysis_text += f"• Male weakest area: {m_weakest} ({gender_means.loc['M', m_weakest]:.2f})\n"
    analysis_text += f"• Female weakest area: {f_weakest} ({gender_means.loc['F', f_weakest]:.2f})"

    # Add text box with analysis
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, 
               bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}, wrap=True)

plt.tight_layout(rect=[0, 0.15, 1, 0.97])  # Adjust layout to make room for text
plt.savefig('output/radar_chart_gender.png', dpi=300)
logger.info("Radar chart by gender saved as 'output/radar_chart_gender.png'")

# Radar chart by experience level
plt.figure(figsize=(12, 12))  # Increased height for analysis text
ax = plt.subplot(111, polar=True)

# Get mean values by experience level
exp_means = df.groupby('Experience Level')[metrics].mean()

# Plot data for each experience level
colors = ['#FF9671', '#845EC2', '#00C2A8', '#FFC75F']
markers = ['o', 's', '^', 'd']
valid_levels = []
for i, level in enumerate(exp_means.index):
    if level != 'Unknown':  # Skip unknown level
        valid_levels.append(level)
        values = exp_means.loc[level].values.flatten().tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linestyle='-', linewidth=2, markersize=8, 
                label=f'{level}', color=colors[i % len(colors)], 
                marker=markers[i % len(markers)])
        ax.fill(angles, values, colors[i % len(colors)], alpha=0.1)

# Customize the chart
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3, 4], ["1", "2", "3", "4"], color="grey", size=10)
plt.ylim(0, 4)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Fitness Profile by Experience Level', fontsize=16, fontweight='bold', y=1.1)

# Prepare analysis text if we have at least two valid experience levels
if len(valid_levels) >= 2:
    # Find metrics with biggest improvement across experience levels
    exp_diff = {}
    for metric in metrics:
        metric_values = exp_means.loc[valid_levels, metric]
        min_val = metric_values.min()
        max_val = metric_values.max()
        min_level = metric_values.idxmin()
        max_level = metric_values.idxmax()
        if min_level != max_level:  # Only consider if there's a difference
            exp_diff[metric] = {
                'improvement': max_val - min_val,
                'from_level': min_level,
                'to_level': max_level,
                'min_val': min_val,
                'max_val': max_val
            }

    # Sort metrics by improvement
    sorted_metrics = sorted(exp_diff.items(), key=lambda x: x[1]['improvement'], reverse=True)

    # Create analysis text
    analysis_text = "ANALYSIS:\n"

    # Add top 3 metrics with most improvement (or fewer if not enough)
    for i, (metric, data) in enumerate(sorted_metrics[:3]):
        analysis_text += f"• {metric}: {data['improvement']:.2f} point improvement from {data['from_level']} to {data['to_level']}\n"

    # Find the most consistent metric across experience levels
    metric_std = exp_means.loc[valid_levels].std()
    most_consistent = metric_std.idxmin()
    analysis_text += f"• Most consistent metric across experience levels: {most_consistent} (std: {metric_std[most_consistent]:.2f})\n"

    # Find which experience level has the highest overall average
    level_avgs = exp_means.loc[valid_levels].mean(axis=1)
    best_level = level_avgs.idxmax()
    analysis_text += f"• Highest overall performance: {best_level} (avg: {level_avgs[best_level]:.2f})"

    # Add text box with analysis
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, 
               bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}, wrap=True)

plt.tight_layout(rect=[0, 0.15, 1, 0.97])  # Adjust layout to make room for text
plt.savefig('output/radar_chart_experience.png', dpi=300)
logger.info("Radar chart by experience level saved as 'output/radar_chart_experience.png'")

# Create a summary dashboard
plt.figure(figsize=(20, 18))  # Increased height for analysis text

# 1. Top left: Gender distribution pie chart
plt.subplot(2, 3, 1)
valid_genders = ['M', 'F']
gender_counts = df[df['Gender'].isin(valid_genders)]['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
        colors=['#FF9671', '#845EC2'], startangle=90)
plt.title('Gender Distribution', fontsize=14, fontweight='bold')

# 2. Top middle: Experience level distribution bar chart
plt.subplot(2, 3, 2)
exp_counts = df['Experience Level'].value_counts().sort_index()
if 'Unknown' in exp_counts.index:
    exp_counts = exp_counts.drop('Unknown')  # Remove unknown level
sns.barplot(x=exp_counts.index, y=exp_counts.values, hue=exp_counts.index, palette='viridis', legend=False)
plt.title('Experience Level Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Count')

# 3. Top right: Total score distribution histogram
plt.subplot(2, 3, 3)
sns.histplot(df['Total'].dropna(), kde=True, bins=15, color='skyblue')
plt.axvline(df['Total'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["Total"].mean():.1f}')
plt.title('Total Score Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Total Score')
plt.ylabel('Count')
plt.legend()

# 4. Bottom left: Average scores by metric bar chart
plt.subplot(2, 3, 4)
avg_scores = df[metrics].mean().sort_values(ascending=False)
sns.barplot(x=avg_scores.index, y=avg_scores.values, hue=avg_scores.index, palette='Blues_d', legend=False)
plt.title('Average Scores by Metric', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Average Score')
for i, v in enumerate(avg_scores):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

# 5. Bottom middle: Correlation mini-heatmap (focused on Total)
plt.subplot(2, 3, 5)
total_corr = correlation['Total'].drop('Total').sort_values(ascending=False)
sns.barplot(x=total_corr.index, y=total_corr.values, hue=total_corr.index, palette='RdBu_r', legend=False)
plt.title('Correlation with Total Score', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='black', linestyle='-', alpha=0.3)
for i, v in enumerate(total_corr):
    plt.text(i, v + 0.02 if v > 0 else v - 0.08, f'{v:.2f}', ha='center', fontweight='bold')

# 6. Bottom right: Gender comparison bar chart
plt.subplot(2, 3, 6)
gender_diff = gender_means.loc['M'] - gender_means.loc['F']
gender_diff = gender_diff.sort_values(ascending=False)
colors = ['#FF9671' if x > 0 else '#845EC2' for x in gender_diff]
sns.barplot(x=gender_diff.index, y=gender_diff.values, hue=gender_diff.index, palette=colors, legend=False)
plt.title('Gender Difference (M - F)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Difference in Average Score')
plt.axhline(0, color='black', linestyle='-', alpha=0.5)
for i, v in enumerate(gender_diff):
    plt.text(i, v + 0.05 if v > 0 else v - 0.15, f'{v:.2f}', ha='center', fontweight='bold')

# Prepare summary analysis text
# Get key statistics
total_participants = len(df)
gender_ratio = f"{gender_counts['M'] / gender_counts.sum() * 100:.1f}% male, {gender_counts['F'] / gender_counts.sum() * 100:.1f}% female"
avg_total = df['Total'].mean()
top_metric = avg_scores.index[0]
bottom_metric = avg_scores.index[-1]
strongest_corr = total_corr.index[0]
biggest_gender_diff = gender_diff.abs().idxmax()
gender_with_advantage = "males" if gender_diff[biggest_gender_diff] > 0 else "females"

# Create analysis text
analysis_text = "SUMMARY ANALYSIS:\n\n"
analysis_text += f"• Total participants: {total_participants} ({gender_ratio})\n"
analysis_text += f"• Average total score: {avg_total:.2f} out of maximum possible score\n"
analysis_text += f"• Strongest performance area: {top_metric} (avg: {avg_scores[top_metric]:.2f})\n"
analysis_text += f"• Weakest performance area: {bottom_metric} (avg: {avg_scores[bottom_metric]:.2f})\n"
analysis_text += f"• {strongest_corr} has the strongest correlation with total score ({total_corr[strongest_corr]:.2f})\n"
analysis_text += f"• Biggest gender difference is in {biggest_gender_diff}, where {gender_with_advantage} perform better\n"

# Add recommendations
analysis_text += "\nKEY RECOMMENDATIONS:\n"
analysis_text += f"1. Focus training programs on improving {bottom_metric} across all participants\n"
analysis_text += f"2. Develop specialized training for {gender_with_advantage=='males' and 'females' or 'males'} to improve {biggest_gender_diff}\n"
analysis_text += f"3. Emphasize {strongest_corr} training as it has the strongest impact on overall performance\n"
analysis_text += f"4. Create targeted programs for beginners focusing on fundamental skills\n"
analysis_text += f"5. Implement regular assessment of {top_metric} to maintain high performance standards"

# Add text box with analysis
plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, 
           bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}, wrap=True)

plt.tight_layout(rect=[0, 0.2, 1, 0.97])  # Adjust layout to make room for text
plt.savefig('output/summary_dashboard.png', dpi=300)
logger.info("Summary dashboard saved as 'output/summary_dashboard.png'")

try:
    # Get unique genders and experience levels
    valid_genders = ['M', 'F']
    genders = df[df['Gender'].isin(valid_genders)]['Gender'].unique()
    experience_levels = df['Experience Level'].unique()

    # Define the metrics to analyze (already defined above)
    # metrics = ['Mobility', 'Core', 'Pullup', 'Bench', 'Pushup', 'Cardio']

    # Create separate visualizations for each gender at each experience level
    for gender in genders:
        for level in experience_levels:
            # Filter data for this gender and experience level
            filtered_df = df[(df['Gender'] == gender) & (df['Experience Level'] == level)]

            # Skip if no data for this combination
            if filtered_df.empty:
                logger.info(f"No data for Gender: {gender}, Experience Level: {level}")
                continue

            # Calculate mean scores for this group
            mean_scores = filtered_df[metrics].mean()

            # Create a figure for this gender and experience level
            plt.figure(figsize=(12, 8))

            # Create a bar chart of mean scores
            ax = mean_scores.plot(kind='bar', color='skyblue', edgecolor='black')

            # Add value labels on top of each bar
            for i, v in enumerate(mean_scores):
                ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontweight='bold')

            # Improve readability
            plt.title(f'Mean Scores for {gender} at Experience Level {level}', fontsize=16, fontweight='bold')
            plt.ylabel('Mean Score', fontsize=14)
            plt.xlabel('Fitness Test', fontsize=14)
            plt.xticks(rotation=30, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add a horizontal line for the average of all metrics
            avg_all = mean_scores.mean()
            plt.axhline(y=avg_all, color='red', linestyle='--', 
                       label=f'Average of all tests: {avg_all:.1f}')

            # Add the number of participants in this group
            plt.figtext(0.5, 0.01, f'Number of participants: {len(filtered_df)}', 
                       ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

            # Add legend
            plt.legend(fontsize=12)

            # Adjust layout and save
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            filename = f'output/fitness_analysis_{gender}_{level}.png'
            plt.savefig(filename, dpi=300)
            logger.info(f"Visualization saved as '{filename}'")

    logger.info("All visualizations have been created successfully.")

    # Add analysis to identify struggle areas for each gender and experience level
    logger.info("=======================================================================")
    logger.info("                STRUGGLE AREA ANALYSIS")
    logger.info("=======================================================================")
    logger.info("Identifying areas where each gender and experience level struggles the most...")

    # Create a DataFrame to store the struggle analysis results
    struggle_results = []

    # Analyze each gender and experience level combination
    valid_genders = ['M', 'F']
    for gender in genders:
        if gender not in valid_genders:  # Skip invalid gender
            continue

        for level in experience_levels:
            if level == 'Unknown':  # Skip unknown level
                continue

            # Filter data for this gender and experience level
            filtered_df = df[(df['Gender'] == gender) & (df['Experience Level'] == level)]

            # Skip if no data for this combination
            if filtered_df.empty:
                continue

            # Calculate mean scores for this group
            mean_scores = filtered_df[metrics].mean()

            # Identify the lowest scoring metric (biggest struggle)
            lowest_metric = mean_scores.idxmin()
            lowest_score = mean_scores.min()

            # Calculate how far below the average this metric is
            avg_all_metrics = mean_scores.mean()
            pct_below_avg = ((avg_all_metrics - lowest_score) / avg_all_metrics) * 100 if avg_all_metrics > 0 else 0

            # Calculate standard deviation to see if the struggle is significant
            std_dev = filtered_df[metrics].std()[lowest_metric]

            # Store the results
            struggle_results.append({
                'Gender': gender,
                'Experience Level': level,
                'Participants': len(filtered_df),
                'Biggest Struggle': lowest_metric,
                'Score': lowest_score,
                'Avg All Metrics': avg_all_metrics,
                'Pct Below Avg': pct_below_avg,
                'Std Dev': std_dev
            })

    # Convert to DataFrame for easier analysis
    struggle_df = pd.DataFrame(struggle_results)

    # Sort by percentage below average to find the most significant struggles
    struggle_df = struggle_df.sort_values('Pct Below Avg', ascending=False)

    # Display the results
    logger.info("Struggle areas by gender and experience level (sorted by significance):")
    pd.set_option('display.max_columns', None)
    logger.info("\n%s", struggle_df)

    # Create a visualization of the struggle areas
    plt.figure(figsize=(14, 12))  # Increased height for analysis text

    # Create a grouped bar chart showing the struggle metric vs. average of all metrics
    groups = struggle_df.apply(lambda x: f"{x['Gender']}-{x['Experience Level']}", axis=1)
    x = np.arange(len(groups))
    width = 0.35

    # Plot bars
    plt.bar(x - width/2, struggle_df['Score'], width, label='Struggle Area Score', color='#FF9671')
    plt.bar(x + width/2, struggle_df['Avg All Metrics'], width, label='Avg of All Metrics', color='#845EC2')

    # Add labels and formatting
    plt.xlabel('Gender-Experience Level', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Struggle Areas by Gender and Experience Level', fontsize=16, fontweight='bold')
    plt.xticks(x, groups, rotation=45, ha='right')
    plt.legend()

    # Add struggle area labels above each bar
    for i, row in enumerate(struggle_df.iterrows()):
        plt.text(i - width/2, row[1]['Score'] - 0.3, row[1]['Biggest Struggle'], 
                ha='center', va='top', rotation=90, color='white', fontweight='bold')
        plt.text(i - width/2, 0.1, f"{row[1]['Pct Below Avg']:.1f}%", 
                ha='center', va='bottom', color='black', fontsize=9)

    # Prepare analysis text
    # Find the most common struggle area across all groups
    struggle_counts = struggle_df['Biggest Struggle'].value_counts()
    most_common_struggle = struggle_counts.idxmax()
    struggle_count = struggle_counts[most_common_struggle]
    total_groups = len(struggle_df)

    # Find the group with the biggest gap between struggle area and average
    struggle_df['Gap'] = struggle_df['Avg All Metrics'] - struggle_df['Score']
    biggest_gap_idx = struggle_df['Gap'].idxmax()
    biggest_gap_group = f"{struggle_df.loc[biggest_gap_idx, 'Gender']}-{struggle_df.loc[biggest_gap_idx, 'Experience Level']}"
    biggest_gap_metric = struggle_df.loc[biggest_gap_idx, 'Biggest Struggle']
    biggest_gap_value = struggle_df.loc[biggest_gap_idx, 'Gap']
    biggest_gap_pct = struggle_df.loc[biggest_gap_idx, 'Pct Below Avg']

    # Find gender-specific patterns
    gender_struggles = {}
    for gender in struggle_df['Gender'].unique():
        gender_data = struggle_df[struggle_df['Gender'] == gender]
        if not gender_data.empty:
            gender_struggle_counts = gender_data['Biggest Struggle'].value_counts()
            if not gender_struggle_counts.empty:
                gender_struggles[gender] = gender_struggle_counts.idxmax()

    # Create analysis text
    analysis_text = "ANALYSIS:\n"
    analysis_text += f"• Most common struggle area: {most_common_struggle} ({struggle_count}/{total_groups} groups)\n"
    analysis_text += f"• Group with biggest performance gap: {biggest_gap_group} in {biggest_gap_metric}\n"
    analysis_text += f"  (Gap: {biggest_gap_value:.1f} points, {biggest_gap_pct:.1f}% below their average)\n"

    # Add gender-specific insights
    analysis_text += "• Gender-specific struggle patterns:\n"
    for gender, struggle in gender_struggles.items():
        analysis_text += f"  - {gender}: Most commonly struggles with {struggle}\n"

    # Add recommendation
    analysis_text += f"• Recommendation: Prioritize {most_common_struggle} training across all groups,\n"
    analysis_text += f"  with special attention to {biggest_gap_group} participants."

    # Add text box with analysis
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, 
               bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}, wrap=True)

    plt.tight_layout(rect=[0, 0.22, 1, 0.97])  # Adjust layout to make room for text
    plt.savefig('output/struggle_areas_analysis.png', dpi=300)
    logger.info("Struggle areas analysis saved as 'output/struggle_areas_analysis.png'")

    # Create a heatmap showing struggle areas across all combinations
    # Pivot the data to create a matrix of gender-experience vs metrics
    logger.info("Generating heatmap of all metrics by gender and experience level...")

    # Create a comprehensive analysis of all metrics
    comprehensive_results = []

    valid_genders = ['M', 'F']
    for gender in genders:
        if gender not in valid_genders:  # Skip invalid gender
            continue

        for level in experience_levels:
            if level == 'Unknown':  # Skip unknown level
                continue

            # Filter data for this gender and experience level
            filtered_df = df[(df['Gender'] == gender) & (df['Experience Level'] == level)]

            # Skip if no data for this combination
            if filtered_df.empty:
                continue

            # Calculate mean scores for this group for each metric
            for metric in metrics:
                mean_score = filtered_df[metric].mean()
                group_avg = filtered_df[metrics].mean().mean()
                pct_diff = ((mean_score - group_avg) / group_avg) * 100 if group_avg > 0 else 0

                comprehensive_results.append({
                    'Gender': gender,
                    'Experience Level': level,
                    'Metric': metric,
                    'Score': mean_score,
                    'Pct Diff From Avg': pct_diff
                })

    # Convert to DataFrame
    comp_df = pd.DataFrame(comprehensive_results)

    # Create a pivot table for the heatmap
    pivot_df = comp_df.pivot_table(
        index=['Gender', 'Experience Level'], 
        columns='Metric', 
        values='Score'
    )

    # Create the heatmap
    plt.figure(figsize=(12, 10))  # Increased height for analysis text
    sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", center=2)
    plt.title('Performance Heatmap by Gender and Experience Level', fontsize=16, fontweight='bold')

    # Prepare analysis text
    # Find highest and lowest values in the heatmap
    max_val = pivot_df.max().max()
    min_val = pivot_df.min().min()
    max_loc = np.where(pivot_df.values == max_val)
    min_loc = np.where(pivot_df.values == min_val)

    # Get the indices for max and min values
    if len(max_loc[0]) > 0:
        max_row_idx = max_loc[0][0]
        max_col_idx = max_loc[1][0]
        max_row = pivot_df.index[max_row_idx]
        max_col = pivot_df.columns[max_col_idx]
    else:
        max_row, max_col = ("Unknown", "Unknown")

    if len(min_loc[0]) > 0:
        min_row_idx = min_loc[0][0]
        min_col_idx = min_loc[1][0]
        min_row = pivot_df.index[min_row_idx]
        min_col = pivot_df.columns[min_col_idx]
    else:
        min_row, min_col = ("Unknown", "Unknown")

    # Calculate average scores by gender and experience level
    gender_avgs = {}
    exp_avgs = {}

    for (gender, exp_level) in pivot_df.index:
        # Add to gender averages
        if gender not in gender_avgs:
            gender_avgs[gender] = []
        gender_avgs[gender].extend(pivot_df.loc[(gender, exp_level)].values)

        # Add to experience level averages
        if exp_level not in exp_avgs:
            exp_avgs[exp_level] = []
        exp_avgs[exp_level].extend(pivot_df.loc[(gender, exp_level)].values)

    # Calculate averages
    for gender in gender_avgs:
        gender_avgs[gender] = sum(gender_avgs[gender]) / len(gender_avgs[gender])
    for exp in exp_avgs:
        exp_avgs[exp] = sum(exp_avgs[exp]) / len(exp_avgs[exp])

    # Find best gender and experience level
    best_gender = max(gender_avgs.items(), key=lambda x: x[1])[0]
    best_exp = max(exp_avgs.items(), key=lambda x: x[1])[0]

    # Create analysis text
    analysis_text = "ANALYSIS:\n"
    analysis_text += f"• Highest performance: {max_row[0]}-{max_row[1]} in {max_col} ({max_val:.1f})\n"
    analysis_text += f"• Lowest performance: {min_row[0]}-{min_row[1]} in {min_col} ({min_val:.1f})\n"
    analysis_text += f"• Best performing gender overall: {best_gender} (avg: {gender_avgs[best_gender]:.1f})\n"
    analysis_text += f"• Best performing experience level: {best_exp} (avg: {exp_avgs[best_exp]:.1f})\n"

    # Find the most challenging metric across all groups
    metric_avgs = pivot_df.mean().sort_values()
    hardest_metric = metric_avgs.index[0]
    easiest_metric = metric_avgs.index[-1]
    analysis_text += f"• Most challenging metric for all groups: {hardest_metric} (avg: {metric_avgs[hardest_metric]:.1f})\n"
    analysis_text += f"• Easiest metric for all groups: {easiest_metric} (avg: {metric_avgs[easiest_metric]:.1f})"

    # Add text box with analysis
    plt.figtext(0.5, 0.01, analysis_text, ha="center", fontsize=12, 
               bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5}, wrap=True)

    plt.tight_layout(rect=[0, 0.2, 1, 0.97])  # Adjust layout to make room for text
    plt.savefig('output/performance_heatmap.png', dpi=300)
    logger.info("Performance heatmap saved as 'output/performance_heatmap.png'")

    # Generate actionable insights
    logger.info("=======================================================================")
    logger.info("                ACTIONABLE INSIGHTS")
    logger.info("=======================================================================")

    # 1. Find the most common struggle area across all groups
    most_common_struggle = comp_df.groupby('Metric')['Score'].mean().idxmin()
    logger.info(f"\n1. The most common struggle area across all groups is: {most_common_struggle}")
    logger.info(f"   Recommendation: Consider developing targeted training programs for {most_common_struggle}.")

    # 2. Identify if certain experience levels consistently struggle with specific metrics
    exp_struggles = comp_df.groupby(['Experience Level', 'Metric'])['Score'].mean().reset_index()
    exp_struggles = exp_struggles.sort_values('Score')

    logger.info("\n2. Experience level-specific struggles:")
    for level in experience_levels:
        if level == 'Unknown':
            continue
        level_data = exp_struggles[exp_struggles['Experience Level'] == level].head(1)
        if not level_data.empty:
            logger.info(f"   {level}: Struggles most with {level_data.iloc[0]['Metric']} (avg score: {level_data.iloc[0]['Score']:.1f})")
            logger.info(f"   Recommendation: Provide {level} participants with additional focus on {level_data.iloc[0]['Metric']} exercises.")

    # 3. Gender-specific insights
    valid_genders = ['M', 'F']
    gender_struggles = comp_df[comp_df['Gender'].isin(valid_genders)].groupby(['Gender', 'Metric'])['Score'].mean().reset_index()
    gender_struggles = gender_struggles.sort_values(['Gender', 'Score'])

    logger.info("\n3. Gender-specific struggles:")
    for gender in valid_genders:
        gender_data = gender_struggles[gender_struggles['Gender'] == gender].head(1)
        if not gender_data.empty:
            logger.info(f"   {gender}: Struggles most with {gender_data.iloc[0]['Metric']} (avg score: {gender_data.iloc[0]['Score']:.1f})")
            logger.info(f"   Recommendation: Develop {gender}-specific training approaches for {gender_data.iloc[0]['Metric']}.")

    # 4. Identify the biggest gap between genders
    gender_pivot = comp_df.pivot_table(index='Metric', columns='Gender', values='Score')
    if 'M' in gender_pivot.columns and 'F' in gender_pivot.columns:
        gender_pivot['Gap'] = gender_pivot['M'] - gender_pivot['F']
        max_gap_metric = gender_pivot['Gap'].abs().idxmax()
        gap_value = gender_pivot.loc[max_gap_metric, 'Gap']

        logger.info(f"\n4. The biggest performance gap between genders is in: {max_gap_metric} (gap: {gap_value:.1f})")
        if gap_value > 0:
            logger.info(f"   Males score higher than females in this metric.")
        else:
            logger.info(f"   Females score higher than males in this metric.")
        logger.info(f"   Recommendation: Address gender-specific challenges in {max_gap_metric} training.")

    # 5. Experience progression insights
    logger.info("\n5. Experience progression insights:")
    for metric in metrics:
        exp_progression = comp_df[comp_df['Metric'] == metric].groupby('Experience Level')['Score'].mean()
        if len(exp_progression) > 1:
            min_exp = exp_progression.idxmin()
            max_exp = exp_progression.idxmax()
            if min_exp != max_exp:
                improvement = exp_progression[max_exp] - exp_progression[min_exp]
                logger.info(f"   {metric}: Shows {improvement:.1f} points improvement from {min_exp} to {max_exp}")

    # Export the struggle analysis to CSV
    struggle_df.to_csv('output/struggle_analysis.csv', index=False)
    logger.info("Struggle analysis exported to 'output/struggle_analysis.csv'")

    logger.info("Actionable insights analysis complete.")

except Exception as e:
    logger.exception(f"An error occurred: {e}")
