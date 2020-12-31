import pymysql

conn = pymysql.connect(host='healrodb.ciovcwimqlrt.us-east-2.rds.amazonaws.com',
                                  user='healro',
                                  password='hongikhealro',
                                  db='healrodb',
                                  charset='utf8',
                                  )

curs = conn.cursor(pymysql.cursors.DictCursor)

sql = "select * from hospital"
curs.execute(sql)

rows = curs.fetchall()


conn.commit()
conn.close()