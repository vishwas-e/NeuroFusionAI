"""
NeuroFusion AI — Database Management & Cleaning Utility
Usage in VS Code Terminal:
    python manage_db.py                  (Interactive Menu)
    python manage_db.py --view           (View all tables)
    python manage_db.py --clear-patients (Delete all patient cases & analyses)
    python manage_db.py --delete-patient "Name" (Delete specific patient)
    python manage_db.py --reset-all      (Reset DB to fresh state)
"""

import sqlite3
import os
import sys

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "neurofusion.db")

def get_connection():
    if not os.path.exists(DB_PATH):
        print(f"[!] Database file not found at: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def view_data():
    conn = get_connection()
    c = conn.cursor()

    print("\n" + "=" * 70)
    print("  1. REGISTERED DOCTORS (users table)")
    print("=" * 70)
    users = c.execute("SELECT id, username, full_name, role FROM users").fetchall()
    if users:
        print(f"{'ID':<4} | {'Username':<12} | {'Doctor Name':<24} | {'Role'}")
        print("-" * 70)
        for u in users:
            print(f"{u['id']:<4} | {u['username']:<12} | {u['full_name']:<24} | {u['role']}")
    else:
        print("  No users found.")

    print("\n" + "=" * 70)
    print("  2. PATIENT RECORDS (patients table)")
    print("=" * 70)
    patients = c.execute("SELECT id, patient_id, age, gender, headache, seizures FROM patients").fetchall()
    if patients:
        print(f"{'ID':<4} | {'Patient Name':<16} | {'Age':<5} | {'Gender':<8} | {'Headache':<9} | {'Seizures'}")
        print("-" * 70)
        for p in patients:
            print(f"{p['id']:<4} | {p['patient_id']:<16} | {str(p['age']):<5} | {str(p['gender']):<8} | {str(p['headache']):<9} | {str(p['seizures'])}")
    else:
        print("  No patient records found.")

    print("\n" + "=" * 70)
    print("  3. DIAGNOSTIC EVALUATIONS (analyses table)")
    print("=" * 70)
    analyses = c.execute("""
        SELECT a.id, a.patient_id, u.full_name as doctor_name, 
               a.predicted_class, a.confidence, a.analysis_date
        FROM analyses a
        LEFT JOIN users u ON a.doctor_id = u.id
        ORDER BY a.id DESC
    """).fetchall()
    if analyses:
        print(f"{'ID':<4} | {'Patient':<16} | {'Doctor':<20} | {'Finding':<12} | {'Confidence':<10} | {'Date'}")
        print("-" * 75)
        for a in analyses:
            conf = f"{a['confidence']*100:.1f}%" if a['confidence'] is not None else "N/A"
            doc = a['doctor_name'] or "Dr. S. Rao, MD"
            print(f"{a['id']:<4} | {a['patient_id']:<16} | {doc:<20} | {str(a['predicted_class']):<12} | {conf:<10} | {a['analysis_date']}")
    else:
        print("  No diagnostic evaluations found.")
    print("\n" + "=" * 70 + "\n")
    conn.close()

def delete_specific_patient(patient_name):
    if not patient_name:
        print("[-] Please specify a patient name.")
        return
    conn = get_connection()
    c = conn.cursor()
    
    # Count matching
    n_pts = c.execute("SELECT COUNT(*) FROM patients WHERE LOWER(patient_id) = LOWER(?)", (patient_name,)).fetchone()[0]
    n_ans = c.execute("SELECT COUNT(*) FROM analyses WHERE LOWER(patient_id) = LOWER(?)", (patient_name,)).fetchone()[0]
    
    if n_pts == 0 and n_ans == 0:
        print(f"[-] No records found for patient: '{patient_name}'")
        conn.close()
        return

    c.execute("DELETE FROM analyses WHERE LOWER(patient_id) = LOWER(?)", (patient_name,))
    c.execute("DELETE FROM patients WHERE LOWER(patient_id) = LOWER(?)", (patient_name,))
    conn.commit()
    conn.close()
    print(f"[✓] Successfully deleted all records for patient: '{patient_name}' ({n_ans} analysis records and {n_pts} profile records removed).")

def clear_all_patients():
    conn = get_connection()
    c = conn.cursor()
    n_ans = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    n_pts = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    c.execute("DELETE FROM analyses")
    c.execute("DELETE FROM patients")
    conn.commit()
    conn.close()
    print(f"[✓] Successfully cleared all patient data ({n_pts} patients, {n_ans} analyses deleted). Doctor user accounts remain intact.")

def reset_all_database():
    from database.database import initialise_database
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[✓] Removed existing database file.")
    initialise_database()
    print("[✓] Database re-initialized cleanly with default doctor (username: 'doctor' / pass: 'neurofusion2024').")

def interactive_menu():
    while True:
        print("\n" + "=" * 50)
        print("  🧠 NeuroFusion AI — Database Manager")
        print("=" * 50)
        print("  [1] View All Data (Doctors, Patients, Analyses)")
        print("  [2] Delete Specific Patient by Name")
        print("  [3] Delete All Patient Cases & Analyses (Keep Doctors)")
        print("  [4] Reset Database Completely (Fresh State)")
        print("  [5] Exit")
        print("-" * 50)
        choice = input("Enter choice (1-5): ").strip()

        if choice == "1":
            view_data()
        elif choice == "2":
            pname = input("Enter patient name to delete: ").strip()
            delete_specific_patient(pname)
        elif choice == "3":
            confirm = input("Are you sure you want to delete ALL patient cases? (y/n): ").strip().lower()
            if confirm == 'y':
                clear_all_patients()
            else:
                print("Cancelled.")
        elif choice == "4":
            confirm = input("Are you sure you want to RESET the entire database? (y/n): ").strip().lower()
            if confirm == 'y':
                reset_all_database()
            else:
                print("Cancelled.")
        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["--view", "-v"]:
            view_data()
        elif arg in ["--clear-patients", "-c"]:
            clear_all_patients()
        elif arg in ["--reset-all", "-r"]:
            reset_all_database()
        elif arg in ["--delete-patient", "-d"] and len(sys.argv) > 2:
            delete_specific_patient(sys.argv[2])
        else:
            print("Unknown argument. Run without arguments for interactive menu:")
            print("  python manage_db.py")
    else:
        interactive_menu()
