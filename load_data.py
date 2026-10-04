import pandas as pd
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# 1. CSV read karein
file_path = r'C:/Users/hp/Desktop/CHURN-RETENTION-ENGINE/Data/telco_churn.csv'
df = pd.read_csv(file_path)

# Columns ke names clean karein
df.columns = [c.strip().replace(' ', '_').replace('-', '_').lower() for c in df.columns]

# 2. Password ko URL encode karein taake '@' symbol URL ko break na kare
raw_password = '123456@sql'
safe_password = quote_plus(raw_password)

# 3. Connection string (psycopg2 explicitly mention kiya hai)
engine = create_engine(f'postgresql+psycopg2://postgres:{safe_password}@localhost:5433/churn_db')

# 4. Data upload karein
df.to_sql('telco_churn', engine, if_exists='replace', index=False)

print("SUCCESS: Sara data PostgreSQL mein chala gaya hai! Total rows:", len(df))