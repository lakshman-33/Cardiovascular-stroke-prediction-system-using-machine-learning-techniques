import pandas as pd
import sqlite3
conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()
# load the data into a Pandas DataFrame
users = pd.read_csv('dd.csv')
# write the data to a sqlite table
users.to_sql('BrainStrokeDatabase', conn, if_exists='append', index = False)
c.execute('''SELECT * FROM BrainStrokeDatabase''').fetchall() # [(1, 'pokerkid'), (2, 'crazyken')]