
import pandas as pd

print("=== BIG FILE (first 3 rows) ===")
df1 = pd.read_csv("data/log_files.csv", nrows=3,sep=None, 
    engine='python', 
    on_bad_lines='skip')
print(df1.dtypes)
print(df1.head(3))

print("\n=== SMALL FILE (first 3 rows) ===")
df2 = pd.read_csv("data/logging_monitoring_anomalies.csv", nrows=3)
print(df2.dtypes)
print(df2.head(3))
