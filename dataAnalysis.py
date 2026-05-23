import pandas as pd
import sqlite3

def analyze():
    conn = sqlite3.connect('GideonLog.db')
    df = pd.read_sql_query("SELECT * from  history", conn)
    print(df.head())
    conn.close()

if __name__ == "__main__":
    analyze()
