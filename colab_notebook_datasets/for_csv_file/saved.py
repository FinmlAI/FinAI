df = pd.DataFrame(flattened_data)
df.to_csv('synthetic_credit_data.csv', index=False)
print("Data saved to synthetic_credit_data.csv")