import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer

conn = sqlite3.connect('GideonLog.db')
df = pd.read_sql_query('SELECT raw_text FROM history', conn)

#this stil doesn't make any sense 
# things to do 
# 1. understand the database for gideon , i.e. sqlite
# 2. add more commands
# 3. understand the ai and make it even robuts

