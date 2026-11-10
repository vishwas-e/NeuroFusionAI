"""
NeuroFusion AI — SQLite Database Layer
=======================================
Handles all persistence: patient records, analysis results,
model metrics, and audit log.

Uses parameterised queries throughout to prevent SQL injection.
"""

import os, sys, json, logging, sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  Connection helper
# ──────────────────────────────────────────────────────────────

@contextmanager
def get_connection(db_path: str = config.DATABASE_PATH):
    """
    Context manager that yields a sqlite3 connection with
    WAL mode and foreign-key enforcement enabled.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row          # row["column"] access
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────
#  Schema initialisation
# ──────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT    NOT NULL UNIQUE,
    age             INTEGER,
    gender          TEXT,
    headache        TEXT,
    vision_problems TEXT,
    seizures        TEXT,
    nausea          TEXT,
    memory_loss     TEXT,
    extra_clinical  TEXT,           -- JSON for any additional fields
    created_at      TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
);

-- Analyses table
CREATE TABLE IF NOT EXISTS analyses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id         TEXT    NOT NULL,
    doctor_id          INTEGER,                -- FK to users.id — owner doctor
    mri_filename       TEXT,
    mri_image_path     TEXT,
    gradcam_image_path TEXT,
    predicted_class    TEXT,
    confidence         REAL,
    probabilities      TEXT,           -- JSON: {"glioma": 0.94, ...}
    analysis_date      TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
    notes              TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (doctor_id)  REFERENCES users(id)
);

-- Model metrics snapshot table
CREATE TABLE IF NOT EXISTS model_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT    NOT NULL,
    accuracy        REAL,
    precision_score REAL,
    recall_score    REAL,
    f1_score        REAL,
    confusion_matrix TEXT,          -- JSON
    recorded_at     TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
);

-- Simple audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event           TEXT    NOT NULL,
    detail          TEXT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
);

-- Registered Clinician Users table
CREATE TABLE IF NOT EXISTS users (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    username       TEXT    NOT NULL UNIQUE,
    password_hash  TEXT    NOT NULL,
    full_name      TEXT,
    role           TEXT    DEFAULT 'Attending Radiologist',
    created_at     TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
);
"""


def hash_password(password: str) -> str:
    """Hash password with SHA-256 and internal salt."""
    import hashlib
    salt = "neurofusion_secure_salt_2024"
    return hashlib.sha256((salt + password.strip()).encode("utf-8")).hexdigest()


def register_user(username: str, password: str, full_name: str = "",
                  role: str = "Attending Radiologist",
                  db_path: str = config.DATABASE_PATH) -> tuple[bool, str]:
    """Register a new clinician user."""
    clean_user = username.strip().lower()
    if not clean_user:
        return False, "Username cannot be empty."
    if len(password.strip()) < 4:
        return False, "Password must be at least 4 characters long."

    hashed = hash_password(password)
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """INSERT INTO users (username, password_hash, full_name, role)
                   VALUES (?, ?, ?, ?)""",
                (clean_user, hashed, full_name.strip() or clean_user.title(), role)
            )
        return True, "Account registered successfully. You can now sign in."
    except sqlite3.IntegrityError:
        return False, f"Username '{clean_user}' is already registered."
    except Exception as exc:
        return False, f"Registration failed: {exc}"


def authenticate_user(username: str, password: str,
                      db_path: str = config.DATABASE_PATH) -> tuple[bool, Optional[dict], str]:
    """Authenticate a clinician user against stored database credentials."""
    clean_user = username.strip().lower()
    hashed = hash_password(password)
    try:
        with get_connection(db_path) as conn:
            row = conn.execute(
                """SELECT id, username, full_name, role FROM users
                   WHERE username = ? AND password_hash = ?""",
                (clean_user, hashed)
            ).fetchone()
            if row:
                return True, dict(row), "Authentication successful."
            else:
                return False, None, "Invalid username or password."
    except Exception as exc:
        return False, None, f"Authentication error: {exc}"


def initialise_database(db_path: str = config.DATABASE_PATH):
    """Create all tables if they do not already exist, and migrate columns if needed."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        # Migrate columns for existing database
        try:
            cols = [r["name"] for r in conn.execute("PRAGMA table_info(analyses);").fetchall()]
            if "mri_image_path" not in cols:
                conn.execute("ALTER TABLE analyses ADD COLUMN mri_image_path TEXT;")
            if "gradcam_image_path" not in cols:
                conn.execute("ALTER TABLE analyses ADD COLUMN gradcam_image_path TEXT;")
            if "doctor_id" not in cols:
                conn.execute("ALTER TABLE analyses ADD COLUMN doctor_id INTEGER;")
        except Exception as e:
            logger.warning("Database migration note: %s", e)

        # Seed default clinician user 'doctor' / 'neurofusion2024' if not present
        try:
            existing = conn.execute("SELECT id FROM users WHERE username = 'doctor'").fetchone()
            if not existing:
                conn.execute(
                    """INSERT INTO users (username, password_hash, full_name, role)
                       VALUES (?, ?, ?, ?)""",
                    ("doctor", hash_password(config.DEMO_PASSWORD), "Dr. S. Rao, MD", "Attending Radiologist")
                )
        except Exception as user_err:
            logger.warning("Default user seed note: %s", user_err)

    logger.info("Database initialised at %s", db_path)


# ──────────────────────────────────────────────────────────────
#  Patient CRUD
# ──────────────────────────────────────────────────────────────

def upsert_patient(patient_info: dict,
                   db_path: str = config.DATABASE_PATH) -> int:
    """
    Insert or update a patient record.

    Parameters
    ----------
    patient_info : dict
        Must contain 'patient_id'.  Other keys map to table columns.

    Returns
    -------
    int  The row id of the inserted/updated patient.
    """
    sql = """
        INSERT INTO patients
            (patient_id, age, gender, headache, vision_problems,
             seizures, nausea, memory_loss, extra_clinical)
        VALUES
            (:patient_id, :age, :gender, :headache, :vision_problems,
             :seizures, :nausea, :memory_loss, :extra_clinical)
        ON CONFLICT(patient_id) DO UPDATE SET
            age             = excluded.age,
            gender          = excluded.gender,
            headache        = excluded.headache,
            vision_problems = excluded.vision_problems,
            seizures        = excluded.seizures,
            nausea          = excluded.nausea,
            memory_loss     = excluded.memory_loss,
            extra_clinical  = excluded.extra_clinical
    """
    row = {
        "patient_id"      : patient_info.get("patient_id", ""),
        "age"             : patient_info.get("age"),
        "gender"          : patient_info.get("gender"),
        "headache"        : patient_info.get("headache"),
        "vision_problems" : patient_info.get("vision_problems"),
        "seizures"        : patient_info.get("seizures"),
        "nausea"          : patient_info.get("nausea"),
        "memory_loss"     : patient_info.get("memory_loss"),
        "extra_clinical"  : json.dumps(patient_info.get("extra_clinical", {})),
    }
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, row)
        return cur.lastrowid


def get_patient(patient_id: str,
                db_path: str = config.DATABASE_PATH) -> Optional[dict]:
    """Fetch a single patient record by patient_id."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
        ).fetchone()
    return dict(row) if row else None


def list_patients(db_path: str = config.DATABASE_PATH) -> list[dict]:
    """Return all patient records ordered by creation date descending."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM patients ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
#  Analysis CRUD
# ──────────────────────────────────────────────────────────────

def save_analysis(patient_id         : str,
                  mri_filename       : str,
                  predicted_class    : str,
                  confidence         : float,
                  probabilities      : dict,
                  notes              : str  = "",
                  mri_image_path     : str  = "",
                  gradcam_image_path : str  = "",
                  doctor_id          : int  = None,
                  db_path            : str  = config.DATABASE_PATH) -> int:
    """
    Persist an analysis result including image file paths.

    Parameters
    ----------
    patient_id         : str   Must match a patient in the patients table.
    mri_filename       : str   Original filename / path reference for the MRI.
    predicted_class    : str
    confidence         : float  (0–1)
    probabilities      : dict   {class_name: probability, ...}
    notes              : str    Optional clinical notes.
    mri_image_path     : str    Local path where the uploaded MRI scan is saved.
    gradcam_image_path : str    Local path where the Grad-CAM heatmap is saved.
    doctor_id          : int    ID of the doctor who performed the analysis.

    Returns
    -------
    int  Row id of the new analysis record.
    """
    sql = """
        INSERT INTO analyses
            (patient_id, doctor_id, mri_filename, mri_image_path, gradcam_image_path,
             predicted_class, confidence, probabilities, notes)
        VALUES
            (:patient_id, :doctor_id, :mri_filename, :mri_image_path, :gradcam_image_path,
             :predicted_class, :confidence, :probabilities, :notes)
    """
    row = {
        "patient_id"         : patient_id,
        "doctor_id"          : doctor_id,
        "mri_filename"       : mri_filename,
        "mri_image_path"     : mri_image_path,
        "gradcam_image_path" : gradcam_image_path,
        "predicted_class"    : predicted_class,
        "confidence"         : round(float(confidence), 6),
        "probabilities"      : json.dumps({k: round(float(v), 6)
                                         for k, v in probabilities.items()}),
        "notes"              : notes,
    }
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, row)
        row_id = cur.lastrowid
    log_event("analysis_saved",
              f"patient={patient_id} class={predicted_class} conf={confidence:.2%} doctor={doctor_id}")
    return row_id



def get_patient_analyses(patient_id: str,
                         doctor_id: Optional[int] = None,
                         db_path: str = config.DATABASE_PATH) -> list[dict]:
    """Return analyses for a given patient, newest first. If doctor_id is specified, filter by doctor."""
    with get_connection(db_path) as conn:
        if doctor_id is not None:
            rows = conn.execute(
                """SELECT a.*, u.full_name as doctor_name, u.role as doctor_role
                   FROM analyses a
                   LEFT JOIN users u ON a.doctor_id = u.id
                   WHERE a.patient_id = ? AND a.doctor_id = ?
                   ORDER BY a.analysis_date DESC""",
                (patient_id, doctor_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT a.*, u.full_name as doctor_name, u.role as doctor_role
                   FROM analyses a
                   LEFT JOIN users u ON a.doctor_id = u.id
                   WHERE a.patient_id = ?
                   ORDER BY a.analysis_date DESC""",
                (patient_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_analysis_by_id(analysis_id: int,
                       db_path: str = config.DATABASE_PATH) -> Optional[dict]:
    """Fetch a single analysis record by its primary key ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT a.*, p.age, p.gender, p.headache, p.vision_problems,
                      p.seizures, p.nausea, p.memory_loss,
                      u.full_name as doctor_name, u.role as doctor_role
               FROM analyses a
               LEFT JOIN patients p ON a.patient_id = p.patient_id
               LEFT JOIN users u ON a.doctor_id = u.id
               WHERE a.id = ?""",
            (analysis_id,),
        ).fetchone()
    return dict(row) if row else None


def list_all_analyses(limit: int = 500,
                      db_path: str = config.DATABASE_PATH) -> list[dict]:
    """Return recent analyses across all patients (admin use)."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT a.*, p.age, p.gender, p.headache, p.vision_problems,
                      p.seizures, p.nausea, p.memory_loss,
                      u.full_name as doctor_name, u.role as doctor_role
               FROM analyses a
               LEFT JOIN patients p ON a.patient_id = p.patient_id
               LEFT JOIN users u ON a.doctor_id = u.id
               ORDER BY a.analysis_date DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_analyses_by_doctor(doctor_id: int,
                            limit: int = 500,
                            db_path: str = config.DATABASE_PATH) -> list[dict]:
    """Return analyses belonging only to the specified doctor."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT a.*, p.age, p.gender, p.headache, p.vision_problems,
                      p.seizures, p.nausea, p.memory_loss,
                      u.full_name as doctor_name, u.role as doctor_role
               FROM analyses a
               LEFT JOIN patients p ON a.patient_id = p.patient_id
               LEFT JOIN users u ON a.doctor_id = u.id
               WHERE a.doctor_id = ?
               ORDER BY a.analysis_date DESC
               LIMIT ?""",
            (doctor_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_analytics_summary(db_path: str = config.DATABASE_PATH) -> dict:
    """Return aggregate statistics across all doctors (admin)."""
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) as n FROM analyses").fetchone()["n"]
        dist_rows = conn.execute(
            """SELECT predicted_class, COUNT(*) as cnt
               FROM analyses GROUP BY predicted_class ORDER BY cnt DESC"""
        ).fetchall()
        avg_conf = conn.execute("SELECT AVG(confidence) as avg_c FROM analyses").fetchone()["avg_c"]
        recent = conn.execute(
            """SELECT patient_id, predicted_class, confidence, analysis_date
               FROM analyses ORDER BY analysis_date DESC LIMIT 5"""
        ).fetchall()
    return {
        "total_cases"       : total,
        "class_distribution": {r["predicted_class"]: r["cnt"] for r in dist_rows},
        "avg_confidence"    : round(avg_conf or 0, 4),
        "recent_analyses"   : [dict(r) for r in recent],
    }


def get_analytics_summary_by_doctor(doctor_id: int,
                                    db_path: str = config.DATABASE_PATH) -> dict:
    """Return aggregate statistics scoped to a single doctor."""
    with get_connection(db_path) as conn:
        total = conn.execute(
            "SELECT COUNT(*) as n FROM analyses WHERE doctor_id = ?", (doctor_id,)
        ).fetchone()["n"]
        dist_rows = conn.execute(
            """SELECT predicted_class, COUNT(*) as cnt
               FROM analyses WHERE doctor_id = ?
               GROUP BY predicted_class ORDER BY cnt DESC""",
            (doctor_id,),
        ).fetchall()
        avg_conf = conn.execute(
            "SELECT AVG(confidence) as avg_c FROM analyses WHERE doctor_id = ?", (doctor_id,)
        ).fetchone()["avg_c"]
        recent = conn.execute(
            """SELECT patient_id, predicted_class, confidence, analysis_date
               FROM analyses WHERE doctor_id = ?
               ORDER BY analysis_date DESC LIMIT 5""",
            (doctor_id,),
        ).fetchall()
    return {
        "total_cases"       : total,
        "class_distribution": {r["predicted_class"]: r["cnt"] for r in dist_rows},
        "avg_confidence"    : round(avg_conf or 0, 4),
        "recent_analyses"   : [dict(r) for r in recent],
    }




# ──────────────────────────────────────────────────────────────
#  Model Metrics
# ──────────────────────────────────────────────────────────────

def save_model_metrics(model_name: str,
                       metrics   : dict,
                       db_path   : str = config.DATABASE_PATH):
    """
    Save evaluation metrics for a trained model snapshot.

    Parameters
    ----------
    model_name : str   e.g. "MRI-only", "Clinical-only", "Fusion"
    metrics    : dict  Keys: accuracy, precision, recall, f1,
                             confusion_matrix (list of lists)
    """
    sql = """
        INSERT INTO model_metrics
            (model_name, accuracy, precision_score, recall_score,
             f1_score, confusion_matrix)
        VALUES
            (:model_name, :accuracy, :precision_score, :recall_score,
             :f1_score, :confusion_matrix)
    """
    with get_connection(db_path) as conn:
        conn.execute(sql, {
            "model_name"    : model_name,
            "accuracy"      : metrics.get("accuracy"),
            "precision_score": metrics.get("precision"),
            "recall_score"  : metrics.get("recall"),
            "f1_score"      : metrics.get("f1"),
            "confusion_matrix": json.dumps(metrics.get("confusion_matrix", [])),
        })
    logger.info("Metrics saved for model: %s", model_name)


def load_latest_metrics(model_name: str,
                        db_path: str = config.DATABASE_PATH) -> Optional[dict]:
    """Load the most recent metric snapshot for a given model."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """SELECT * FROM model_metrics
               WHERE model_name = ?
               ORDER BY recorded_at DESC
               LIMIT 1""",
            (model_name,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["confusion_matrix"] = json.loads(d.get("confusion_matrix") or "[]")
    return d


# ──────────────────────────────────────────────────────────────
#  Audit Log
# ──────────────────────────────────────────────────────────────

def log_event(event: str, detail: str = "",
              db_path: str = config.DATABASE_PATH):
    """Append an entry to the audit log (best-effort; never raises)."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO audit_log (event, detail) VALUES (?, ?)",
                (event, detail),
            )
    except Exception as exc:
        logger.warning("Audit log failed: %s", exc)
