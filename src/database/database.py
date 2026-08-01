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


from datetime import datetime


def save_assessment(features, result):

    print("save_assessment() called")

    conn = get_connection()

    conn.execute("""
        INSERT INTO assessments(
            created_at,
            age_months,
            sex,
            weight,
            height,
            bmi,
            overall_risk,
            underweight,
            stunting,
            wasting
        )
        VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (

        datetime.now().strftime("%Y-%m-%d %H:%M"),

        features["age_months"],

        "Male" if features["sex"] == 1 else "Female",

        features["weight_kg"],

        features["height_cm"],

        result["WHO Growth"]["bmi"],

        result["Overall Risk"],

        int(result["Prediction"]["Underweight"]),

        int(result["Prediction"]["Stunting"]),

        int(result["Prediction"]["Wasting"])

    ))

    conn.commit()
    conn.close()    

def get_all_assessments():

    conn = get_connection()

    cursor = conn.execute("""

        SELECT *

        FROM assessments

        ORDER BY id DESC

    """)

    rows = cursor.fetchall()

    conn.close()

    return rows