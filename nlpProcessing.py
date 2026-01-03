import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer

conn = sqlite3.connect('GideonLog.db')
curs = conn.cursor()


