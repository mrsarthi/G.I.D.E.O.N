import pandas as pd
import sqlite3

conn = sqlite3.connect('GideonLog.db')
cursor = conn.cursor()

# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())

df = pd.read_sql_query("SELECT * from  history", conn)

# print(df.info())
print(df.head())