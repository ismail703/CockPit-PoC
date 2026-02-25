import sqlite3

# Connect to database
conn = sqlite3.connect("your_file.db")

# Create cursor
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())