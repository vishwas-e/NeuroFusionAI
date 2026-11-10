"""
NeuroFusion AI™ — Database Viewer Utility
Run this script in your VS Code terminal to inspect all tables and data:
    python view_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "neurofusion.db")

def print_separator(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)

def view_database():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Registered Clinicians / Users
    print_separator("1. REGISTERED DOCTORS & CLINICIANS (users table)")
    users = cursor.execute("SELECT id, username, full_name, role, created_at FROM users").fetchall()
    if users:
        print(f"{'ID':<4} | {'Username':<12} | {'Doctor Name':<22} | {'Role':<24}")
        print("-" * 65)
        for u in users:
            print(f"{u['id']:<4} | {u['username']:<12} | {u['full_name']:<22} | {u['role']:<24}")
    else:
        print("No users found.")

    # 2. Patients Table
    print_separator("2. PATIENT RECORDS (patients table)")
    patients = cursor.execute("SELECT id, patient_id, age, gender, headache, vision_problems, seizures FROM patients").fetchall()
    if patients:
        print(f"{'ID':<4} | {'Patient Name':<12} | {'Age':<5} | {'Gender':<8} | {'Headache':<9} | {'Vision':<8} | {'Seizures':<8}")
        print("-" * 65)
        for p in patients:
            print(f"{p['id']:<4} | {p['patient_id']:<12} | {str(p['age']):<5} | {str(p['gender']):<8} | {str(p['headache']):<9} | {str(p['vision_problems']):<8} | {str(p['seizures']):<8}")
    else:
        print("No patient records found.")

    # 3. Diagnostic Analyses (with Doctor association)
    print_separator("3. DIAGNOSTIC EVALUATIONS & HISTORIES (analyses table)")
    analyses = cursor.execute("""
        SELECT a.id, a.patient_id, a.doctor_id, u.full_name as doctor_name, 
               a.predicted_class, a.confidence, a.analysis_date
        FROM analyses a
        LEFT JOIN users u ON a.doctor_id = u.id
        ORDER BY a.id DESC
    """).fetchall()
    if analyses:
        print(f"{'ID':<4} | {'Patient':<10} | {'Doctor':<20} | {'Pathology':<12} | {'Confidence':<10} | {'Date'}")
        print("-" * 75)
        for a in analyses:
            conf_str = f"{a['confidence']*100:.1f}%" if a['confidence'] is not None else "N/A"
            doc_str = a['doctor_name'] if a['doctor_name'] else f"Doctor ID {a['doctor_id']}"
            print(f"{a['id']:<4} | {a['patient_id']:<10} | {doc_str:<20} | {str(a['predicted_class']):<12} | {conf_str:<10} | {a['analysis_date']}")
    else:
        print("No analysis evaluations found.")

    print("\n" + "=" * 65 + "\n")
    conn.close()

if __name__ == "__main__":
    view_database()
