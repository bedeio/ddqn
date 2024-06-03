import pandas as pd

TOP_N = 2
# Load the CSV file
file_path = 'cartpole_robust_latest.csv'
df = pd.read_csv(file_path)

# Function to get the top results for each combination of wind, turbulence, and model_type
def get_top_n_results(df):
    df = df.sort_values(by=['wind_power', 'avg_reward', 'avg_time'], ascending=[True, False, True])
    df = df.groupby(['wind_power', 'model_type']).apply(lambda x: x.head(TOP_N)).reset_index(drop=True)
    return df.sort_values(by=['wind_power', 'avg_reward', 'avg_time'], ascending=[True, False, True])

# Apply the function to get the top results
top_n_results = get_top_n_results(df)#.drop(['env_name'], axis=1)

# Save the results to a new CSV file if needed
top_n_results.to_csv(f'top_{TOP_N}_results_robust.csv', index=False)

# Convert the dataframes to LaTeX tables
latex = top_n_results.to_latex(index=False)

# Save LaTeX tables to files
print(f"=== {file_path} ===")
print(latex)