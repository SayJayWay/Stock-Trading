# -*- coding: utf-8 -*-

import sqlite3 as sq3

con = sq3.connect('numbs.db')

def create_Table(): # Only need to do this the first time

    query = 'CREATE TABLE numbs (Date date, No1 real, No2 real)'

    #con.execute(query)
    con.commit()

#q = con.execute # This is to just create a shorter alias to execute commands
#q('SELECT * FROM sqlite_master').fetchall()
    
from datetime import datetime as dt
import numpy as np

now = dt.now()
np.random.seed(100)

data = np.random.standard_normal((10000,2)).round(4)

for row in data:
    now = dt.now()
    q('INSERT INTO numbs VALUES(?,?,?)', (now, row[0], row[1]))
con.commit()

q('SELECT * FROM numbs').fetchmany(4) # need to input this in the console to see results
q('SELECT * FROM numbs WHERE no1 > 0.5').fetchmany(4)
pointer = q('SELECT * FROM numbs')
for i in range(3):
    print(pointer.fetchone())
    
rows = pointer.fetchall()
rows[:3]

q('DROP TABLE IF EXISTS numbs')

q('SELECT * FROM sqlite_master').fetchall() # should return blank if drop command was executed

con.close()
!rm -f $path* # removes database file from disk