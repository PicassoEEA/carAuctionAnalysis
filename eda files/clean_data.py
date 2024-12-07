import pandas as pd

file_path = 'car_prices.csv' 

try:
    df = pd.read_csv(file_path, on_bad_lines='skip')
except pd.errors.ParserError as e:
    print(f"Error parsing file: {e}")

df.to_csv('car_prices_corrected.csv', index=False)