import pandas as pd
import sqlite3
from pathlib import Path
import os

# ---- CONFIG ----
db_path = "cockpit.db"
csv_folder = Path(r"D:\Users\ismail_elmain\Downloads\Cockpit") # Added 'r' for raw string

# ---- CONNECT TO SQLITE ----
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# ---- CREATE TABLE SCHEMA ----
create_table_sql = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id_date   DATETIME,
    id_day    INTEGER,
    id_week   INTEGER,
    id_month  INTEGER,
    id_year   INTEGER,
    valeur_d1    REAL,
    split        TEXT,
    segment      TEXT,
    kpi          TEXT
);
"""

table_names = ["bda_activations", "bda_encaissement", "bda_ftth_delai_raccordement",
               "bda_ftth_reclamations", "bda_parc", "bda_revenu_billing", "bda_recharge"]

# ---- LOOP TO CREATE & SEED TABLES ----
for table_name in table_names:
    try:
        # 1. Force table creation
        cursor.execute(create_table_sql.format(table_name=table_name))
        conn.commit() # Commit schema immediately
        
        csv_file = csv_folder / f"{table_name}_mock.csv"
        
        # 2. Check if file exists before reading
        if not csv_file.exists():
            print(f"⚠️ Skipping {table_name}: File {csv_file} not found.")
            continue

        df = pd.read_csv(csv_file, sep=';')

        # 3. Clean and Validate
        df["id_date"] = pd.to_datetime(df["id_date"], errors="coerce")
        
        # 4. Use 'append' safely
        df.to_sql(table_name, conn, if_exists="append", index=False)
        print(f"✅ {table_name} created and seeded with {len(df)} rows.")

    except Exception as e:
        print(f"❌ Error processing {table_name}: {e}")

# ---- CLOSE ----
conn.close()
print("\n🎉 Process finished. Check the logs above for any skipped tables.")