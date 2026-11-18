import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('A Questionnaire regarding political endorsement in Hellu.csv')

# Map the ordinal social media usage responses to numerical values
sm_mapping = {
    'Less than 1 hour a day': 1,
    '1-2 hours a day': 2,
    '2-4 hours a day': 3,
    '4-6 hours a day': 4,
    'More than 6 hours a day': 5
}
df['Social Media Usage (Ordinal)'] = df['3. How much do you use social media?'].map(sm_mapping)

# Define the columns to include in the correlation matrix
cols_for_corr = [
    '6. How interested are you in politics?',
    '7. How much do you trust politicians?',
    '9. From a purely economic standpoint, would you consider yourself to be',
    '10. From a purely valuational standpoint, would you consider yourself to be',
    'Social Media Usage (Ordinal)'
]

# Rename the columns for a cleaner visualization
rename_dict = {
    '6. How interested are you in politics?': 'Interest in Politics',
    '7. How much do you trust politicians?': 'Trust in Politicians',
    '9. From a purely economic standpoint, would you consider yourself to be': 'Economic Standpoint (Left-Right)',
    '10. From a purely valuational standpoint, would you consider yourself to be': 'Valuational Standpoint (Lib-Con)',
    'Social Media Usage (Ordinal)': 'Social Media Usage'
}
corr_df = df[cols_for_corr].rename(columns=rename_dict)

# Calculate the Spearman correlation matrix
corr_matrix = corr_df.corr(method='spearman')

# Generate and display the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f', square=True)
plt.title('Spearman Correlation Matrix of Political & Media Habits')
plt.tight_layout()
plt.savefig('spearman_correlation.png')
plt.show()
