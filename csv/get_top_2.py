import pandas as pd

# Load the CSV file
file_path = 'cartpole_robust_latest.csv'
df = pd.read_csv(file_path)

# Clean the data by removing rows with non-finite values in 'max_depth' and convert 'max_depth' to int
df_cleaned = df.dropna(subset=['max_depth'])
df_cleaned['max_depth'] = df_cleaned['max_depth'].astype(int)

# Define a function to get the top 2 results for each combination of env_name and model_type
def get_top_results(df):
    return df.sort_values(by=['avg_reward', 'avg_time'], ascending=[False, True]).groupby(['env_name', 'model_type']).head(2)

# Get the top 2 results for each combination
top_results = get_top_results(df_cleaned)

# Split the dataframe by 'env_name'
df_lunar_lander = top_results[top_results['env_name'] == 'LunarLander-v2'].drop(columns=['env_name'])
df_taxi = top_results[top_results['env_name'] == 'Taxi-v3'].drop(columns=['env_name'])
df_cartpole = top_results[top_results['env_name'] == 'CartPole-v1'].drop(columns=['env_name'])

# Convert the dataframes to LaTeX tables
latex_lunar_lander = df_lunar_lander.to_latex(index=False)
latex_taxi = df_taxi.to_latex(index=False)
latex_cartpole = df_cartpole.to_latex(index=False)

# Save LaTeX tables to files
print("=== CART POLE ===")
print(latex_cartpole)
print("=== LUNAR LANDER ===")
print(latex_lunar_lander)
print("=== TAXI ===")
print(latex_taxi)
