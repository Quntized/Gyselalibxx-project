import pandas as pd
file_list = ["full_trajectory_dataset.csv", "full_trajectory_dataset_2.csv"]
master_df = pd.DataFrame()
for file in file_list:
    df = pd.read_csv(file)
    print(df['shot'].min())
    print(df['shot'].max())
    print(f"{file} min shot: {df['shot'].min()}")
    print(f"{file} max shot: {df['shot'].max()}")
    if (file == 'full_trajectory_dataset_2.csv'):
        df['shot'] = df['shot'] + 1000
    master_df = pd.concat([master_df, df], ignore_index = True)
output_name = "master_trajectory_dataset.csv"
master_df.to_csv(output_name, index=False)
