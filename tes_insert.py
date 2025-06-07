import pymysql
from datetime import datetime, time 

conn = pymysql.connect(host='ipdnkalbar.ac.id', user='u1049330_unitit', password='4dumK4lba12', database='u1049330_miot')

def insert_data(t, c):
    cursor = conn.cursor()
    sql = "INSERT INTO traffic_data (timestamp, count) VALUES (%s, %s)"
    val = (t, c)
    cursor.execute(sql, val)
    conn.commit()
    #print(cursor.rowcount, "record inserted.")

timestamp_str = datetime.now()
avg_count = [4, 5, 2, 3, 6, 7, 9, 7, 6, 7, 8, 9]
average = sum(avg_count) / len(avg_count)
insert_data(timestamp_str, average)