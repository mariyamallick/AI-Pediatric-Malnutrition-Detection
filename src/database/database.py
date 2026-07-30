import sqlite3
from pathlib import Path

DB_PATH = Path("database.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_table():
    conn = get_connection()

    conn.execute("""
    CREATE TABLE IF NOT EXISTS assessments(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        created_at TEXT,

        age_months REAL,

        sex TEXT,

        weight REAL,

        height REAL,

        bmi REAL,

        overall_risk TEXT,

        underweight INTEGER,

        stunting INTEGER,

        wasting INTEGER

    )
    """)

    conn.commit()
    conn.close()