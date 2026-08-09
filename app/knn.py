import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database.database import create_table, get_all_assessments


from flask import Flask, render_template, request, send_from_directory
from src.pipeline.prediction_pipeline import assess_child
from src.reports.report_generator import generate_pdf_report
