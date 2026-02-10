import sqlite3

conn = sqlite3.connect("attendance.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT
)
""")

conn.commit()
conn.close()

print("Database created successfully!")
