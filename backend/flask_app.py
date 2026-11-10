"""
NeuroFusion AI™ — Clinical Radiology Workstation (Flask Edition)
================================================================
Independent Flask web application running on local URL (http://127.0.0.1:5000).
Menu Bar: Home Page | New Intake | Patient Information | Analysis
"""

import os, sys, io, json, logging
from datetime import datetime

# ── Windows TF DLL fix: load TF's bundled VC++ 14.50 runtime before System32's
if sys.platform == "win32":
    _tf_py = os.path.join(
        os.path.dirname(sys.executable),
        "Lib", "site-packages", "tensorflow", "python"
    )
    if os.path.isdir(_tf_py):
        os.add_dll_directory(_tf_py)

_local_keras = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".keras")
os.makedirs(_local_keras, exist_ok=True)
os.environ.setdefault("KERAS_HOME", _local_keras)

import cv2
import numpy as np
from PIL import Image
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_from_directory
)

# Project root (parent directory of backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from database.database import (
    initialise_database, authenticate_user, register_user,
    upsert_patient, save_analysis, list_all_analyses,
    list_analyses_by_doctor, get_analytics_summary_by_doctor,
    get_patient, get_analysis_by_id, get_analytics_summary
)
from preprocessing.image_preprocessing import (
    load_and_preprocess_image, load_image_for_display
)
from preprocessing.clinical_preprocessing import ClinicalPreprocessor
from models.model_utils import (
    load_model_safe, load_class_names,
    build_mri_feature_model, build_clinical_feature_model
)
from explainability.gradcam import generate_gradcam
from explainability.diagnostic_report import generate_diagnostic_explanation
from explainability.tumor_segmentation import generate_tumor_segmentation
from utils.helpers import probabilities_to_dict, sanitise_patient_id, safe_json_loads

# ── Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("NeuroFusionFlask")

# ── Flask Application
app = Flask(__name__, template_folder=config.TEMPLATES_DIR)
app.secret_key = "neurofusion_secure_production_key_2024"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# ── Ensure media directories exist
MEDIA_DIR = os.path.join(config.PROCESSED_DIR, "saved_analyses")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ── Global Model Store
MODELS = {
    "mri": None,
    "fusion": None,
    "preprocessor": None,
    "class_names": config.DEFAULT_CLASSES,
}


def init_models():
    """Load neural network checkpoints into memory."""
    logger.info("Initializing neural network checkpoints …")
    try:
        MODELS["mri"] = load_model_safe(config.MRI_MODEL_PATH)
        MODELS["fusion"] = load_model_safe(config.FUSION_MODEL_PATH)
        MODELS["class_names"] = load_class_names()
        if os.path.exists(config.SCALER_PATH) and os.path.exists(config.LABEL_ENCODER_PATH):
            MODELS["preprocessor"] = ClinicalPreprocessor.load()
        logger.info("Models initialized successfully. MRI: %s, Fusion: %s",
                    MODELS["mri"] is not None, MODELS["fusion"] is not None)
    except Exception as exc:
        logger.exception("Failed to initialize models: %s", exc)


# ── Custom Jinja Filters
@app.template_filter("basename")
def basename_filter(path_str):
    if not path_str:
        return ""
    return os.path.basename(path_str.replace("\\", "/"))


# ── Media Server Route
@app.route("/media/<filename>")
def media_file(filename):
    """Serve saved MRI scans and Grad-CAM overlays from the processed media dir."""
    return send_from_directory(MEDIA_DIR, filename)


# ── Authentication Helper
def login_required(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            flash("Please sign in to access the clinical workstation.", "warning")
            return redirect(url_for("login_view"))
        return func(*args, **kwargs)
    return wrapper


# ──────────────────────────────────────────────────────────────
#  Authentication Routes
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if session.get("user"):
        return redirect(url_for("home_view"))
    return redirect(url_for("login_view"))


@app.route("/login", methods=["GET"])
def login_view():
    if session.get("user"):
        return redirect(url_for("home_view"))
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    ok, user_data, msg = authenticate_user(username, password)
    if ok and user_data:
        session["user"] = user_data
        flash(f"Welcome, {user_data.get('full_name', username)}.", "success")
        return redirect(url_for("home_view"))
    else:
        flash(msg or "Invalid username or password.", "error")
        return redirect(url_for("login_view"))


@app.route("/register", methods=["POST"])
def register_post():
    full_name = request.form.get("full_name", "").strip()
    role = request.form.get("role", "Attending Radiologist").strip()
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()
    confirm_password = request.form.get("confirm_password", "").strip()

    if not username:
        flash("Username is required.", "error")
        return redirect(url_for("login_view"))

    if len(password) < 4:
        flash("Password must be at least 4 characters long.", "error")
        return redirect(url_for("login_view"))

    if password != confirm_password:
        flash("Passwords do not match.", "error")
        return redirect(url_for("login_view"))

    ok, msg = register_user(username, password, full_name, role)
    if ok:
        flash(msg + " You can now sign in.", "success")
    else:
        flash(msg, "error")

    return redirect(url_for("login_view"))


@app.route("/logout")
def logout():
    session.clear()
    flash("You have signed out successfully.", "info")
    return redirect(url_for("login_view"))


# ──────────────────────────────────────────────────────────────
#  1. Home Page Route
# ──────────────────────────────────────────────────────────────

@app.route("/home", methods=["GET"])
@login_required
def home_view():
    doctor_id = session["user"]["id"]
    try:
        summary = get_analytics_summary_by_doctor(doctor_id)
        total_cases = summary.get("total_cases", 0)
    except Exception:
        total_cases = 0

    recent_cases = list_analyses_by_doctor(doctor_id, limit=5)
    return render_template(
        "home.html",
        active_page="home",
        total_cases=total_cases,
        recent_cases=recent_cases
    )


# ──────────────────────────────────────────────────────────────
#  2. New Intake Route
# ──────────────────────────────────────────────────────────────

@app.route("/intake", methods=["GET"])
@login_required
def intake_view():
    reanalyze_data = session.pop("reanalyze_data", None)
    return render_template(
        "intake.html",
        active_page="intake",
        reanalyze_data=reanalyze_data
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict_post():
    # ── Verify uploaded file
    file = request.files.get("mri_file")
    if not file or file.filename == "":
        flash("Please upload a brain MRI scan.", "error")
        return redirect(url_for("intake_view"))

    # ── Patient info extraction
    patient_id = sanitise_patient_id(request.form.get("patient_id", "").strip())
    if not patient_id:
        flash("Patient Name is required.", "error")
        return redirect(url_for("intake_view"))

    try:
        age = int(request.form.get("age", 35))
    except ValueError:
        age = 35

    patient_info = {
        "patient_id": patient_id,
        "age": age,
        "gender": request.form.get("gender", "Other"),
        "headache": request.form.get("headache", "No"),
        "vision_problems": request.form.get("vision_problems", "No"),
        "seizures": request.form.get("seizures", "No"),
        "nausea": request.form.get("nausea", "No"),
        "memory_loss": request.form.get("memory_loss", "No"),
    }
    notes = request.form.get("notes", "").strip()

    # ── Inference Pipeline
    mri_bytes = file.read()
    try:
        preprocessed = load_and_preprocess_image(io.BytesIO(mri_bytes))
        original_img = load_image_for_display(io.BytesIO(mri_bytes))
    except Exception as img_err:
        flash(f"Image processing error: {img_err}", "error")
        return redirect(url_for("intake_view"))

    mri_model = MODELS["mri"]
    fusion_model = MODELS["fusion"]
    preprocessor = MODELS["preprocessor"]
    class_names = MODELS["class_names"]

    if mri_model is None:
        flash("MRI inference checkpoint not loaded.", "error")
        return redirect(url_for("intake_view"))

    use_fusion = (fusion_model is not None and preprocessor is not None)

    try:
        if use_fusion:
            mri_feat_model = build_mri_feature_model()
            clin_feat_model = build_clinical_feature_model(preprocessor.feature_dim)
            mri_feat = mri_feat_model.predict(np.expand_dims(preprocessed, 0), verbose=0)
            clin_vec = preprocessor.transform(patient_info)
            clin_feat = clin_feat_model.predict(clin_vec.reshape(1, -1), verbose=0)
            probs = fusion_model.predict([mri_feat, clin_feat], verbose=0)[0]
            model_used = "Multimodal Deep Feature Fusion"
        else:
            probs = mri_model.predict(np.expand_dims(preprocessed, 0), verbose=0)[0]
            model_used = "MRI-Only Deep Convolutional Architecture"
    except Exception as pred_err:
        flash(f"Diagnostic evaluation failed: {pred_err}", "error")
        return redirect(url_for("intake_view"))

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"
    prob_dict = probabilities_to_dict(probs, class_names)

    # ── Grad-CAM Saliency & Colored Tumor Segmentation Generation
    gradcam_overlay = None
    seg_overlay = None
    seg_info = {}
    try:
        cam_res = generate_gradcam(
            model=mri_model,
            preprocessed=preprocessed,
            original_img=original_img,
            class_idx=pred_idx,
        )
        gradcam_overlay = cam_res.get("overlay")
        heatmap = cam_res.get("heatmap")

        if heatmap is not None and original_img is not None:
            seg_res = generate_tumor_segmentation(
                heatmap=heatmap,
                original_img=original_img,
                pred_class=pred_class
            )
            seg_overlay = seg_res.get("overlay")
            seg_info = {
                "detected": seg_res.get("detected"),
                "area_pct": seg_res.get("area_pct"),
                "color_name": seg_res.get("color_name"),
                "message": seg_res.get("message"),
            }
    except Exception as cam_err:
        logger.warning("Explainability generation warning: %s", cam_err)

    # ── Persist images for longitudinal records
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_pid = patient_id.replace("/", "_").replace("\\", "_")
    mri_img_name = f"{safe_pid}_{timestamp_str}_mri.png"
    gradcam_img_name = f"{safe_pid}_{timestamp_str}_gradcam.png"
    seg_img_name = f"{safe_pid}_{timestamp_str}_segmented.png"
    mri_save_path = os.path.join(MEDIA_DIR, mri_img_name)
    gradcam_save_path = os.path.join(MEDIA_DIR, gradcam_img_name)
    seg_save_path = os.path.join(MEDIA_DIR, seg_img_name)

    if isinstance(original_img, np.ndarray):
        cv2.imwrite(mri_save_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    if isinstance(gradcam_overlay, np.ndarray):
        cv2.imwrite(gradcam_save_path, cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
    if isinstance(seg_overlay, np.ndarray):
        cv2.imwrite(seg_save_path, cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR))

    # ── Save to Database
    try:
        doctor_id = session.get("user", {}).get("id")
        upsert_patient(patient_info)
        row_id = save_analysis(
            patient_id=patient_id,
            mri_filename=file.filename,
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=prob_dict,
            notes=notes,
            mri_image_path=mri_save_path,
            gradcam_image_path=gradcam_save_path,
            doctor_id=doctor_id,
        )
    except Exception as db_err:
        logger.warning("Database record save note: %s", db_err)
        row_id = None

    explanation = generate_diagnostic_explanation(
        pred_class=pred_class,
        confidence=confidence,
        patient_info=patient_info,
        probabilities=prob_dict,
    )

    result_data = {
        "id": row_id,
        "predicted_class": pred_class,
        "confidence": confidence,
        "probabilities": prob_dict,
        "model_used": model_used,
        "mri_img_name": mri_img_name,
        "gradcam_img_name": gradcam_img_name,
        "seg_img_name": seg_img_name,
        "seg_info": seg_info,
        "patient_info": patient_info,
        "doctor_name": session.get("user", {}).get("full_name"),
        "doctor_role": session.get("user", {}).get("role"),
        "explanation": explanation,
    }

    session["last_result"] = result_data
    flash("Diagnostic AI evaluation successfully executed.", "success")
    return redirect(url_for("analysis_view"))


# ──────────────────────────────────────────────────────────────
#  3. Patient Information Route (Doctor-Scoped)
# ──────────────────────────────────────────────────────────────

@app.route("/patients", methods=["GET"])
@login_required
def patients_view():
    search = request.args.get("search", "").strip()
    filter_class = request.args.get("filter_class", "").strip()
    doctor_id = session["user"]["id"]

    # Only retrieve patients/analyses for the logged-in doctor
    all_records = list_analyses_by_doctor(doctor_id, limit=500)
    available_classes = sorted(list({a.get("predicted_class", "") for a in all_records if a.get("predicted_class")}))

    filtered = all_records
    if search:
        filtered = [a for a in filtered if search.lower() in (a.get("patient_id") or "").lower()]
    if filter_class:
        filtered = [a for a in filtered if a.get("predicted_class") == filter_class]

    return render_template(
        "patients.html",
        active_page="patients",
        analyses=filtered,
        available_classes=available_classes,
        search_query=search,
        filter_class=filter_class
    )


@app.route("/reanalyze/<patient_id>")
@login_required
def reanalyze(patient_id):
    patient = get_patient(patient_id)
    if patient:
        session["reanalyze_data"] = patient
        flash(f"Preloaded historical data for patient {patient_id}. Upload a scan sequence to evaluate.", "info")
    else:
        session["reanalyze_data"] = {"patient_id": patient_id}
    return redirect(url_for("intake_view"))


# ──────────────────────────────────────────────────────────────
#  4. Analysis Route
# ──────────────────────────────────────────────────────────────

@app.route("/analysis", methods=["GET"])
@login_required
def analysis_view():
    analysis_id = request.args.get("id")
    result = None
    doctor_id = session["user"]["id"]

    # All evaluations by this doctor for the patient switcher
    all_analyses = list_analyses_by_doctor(doctor_id, limit=500)

    if analysis_id:
        try:
            record = get_analysis_by_id(int(analysis_id))
            if record:
                probs = safe_json_loads(record.get("probabilities", "{}"), {})
                mri_name = os.path.basename(record.get("mri_image_path", "").replace("\\", "/"))
                cam_name = os.path.basename(record.get("gradcam_image_path", "").replace("\\", "/"))
                result = {
                    "id": record.get("id"),
                    "predicted_class": record.get("predicted_class", "Unknown"),
                    "confidence": record.get("confidence", 0.0),
                    "probabilities": probs,
                    "model_used": "Multimodal Deep Feature Fusion",
                    "mri_img_name": mri_name,
                    "gradcam_img_name": cam_name,
                    "doctor_name": record.get("doctor_name"),
                    "doctor_role": record.get("doctor_role"),
                    "patient_info": {
                        "patient_id": record.get("patient_id"),
                        "age": record.get("age"),
                        "gender": record.get("gender"),
                        "headache": record.get("headache", "No"),
                        "vision_problems": record.get("vision_problems", "No"),
                        "seizures": record.get("seizures", "No"),
                        "nausea": record.get("nausea", "No"),
                        "memory_loss": record.get("memory_loss", "No"),
                    }
                }
        except Exception as exc:
            logger.warning("Could not fetch analysis by ID: %s", exc)

    if not result:
        # Check session for freshly submitted analysis
        result = session.get("last_result")

    if not result and all_analyses:
        # Load the most recent analysis by this doctor from DB
        latest = all_analyses[0]
        probs = safe_json_loads(latest.get("probabilities", "{}"), {})
        mri_name = os.path.basename(latest.get("mri_image_path", "").replace("\\", "/"))
        cam_name = os.path.basename(latest.get("gradcam_image_path", "").replace("\\", "/"))
        result = {
            "id": latest.get("id"),
            "predicted_class": latest.get("predicted_class", "Unknown"),
            "confidence": latest.get("confidence", 0.0),
            "probabilities": probs,
            "model_used": "Multimodal Deep Feature Fusion",
            "mri_img_name": mri_name,
            "gradcam_img_name": cam_name,
            "doctor_name": latest.get("doctor_name"),
            "doctor_role": latest.get("doctor_role"),
            "patient_info": {
                "patient_id": latest.get("patient_id"),
                "age": latest.get("age"),
                "gender": latest.get("gender"),
                "headache": latest.get("headache", "No"),
                "vision_problems": latest.get("vision_problems", "No"),
                "seizures": latest.get("seizures", "No"),
                "nausea": latest.get("nausea", "No"),
                "memory_loss": latest.get("memory_loss", "No"),
            }
        }

    if result:
        cam_name = result.get("gradcam_img_name", "")
        mri_name = result.get("mri_img_name", "")
        seg_name = cam_name.replace("_gradcam.png", "_segmented.png") if cam_name else None
        seg_path = os.path.join(MEDIA_DIR, seg_name) if seg_name else ""
        mri_path = os.path.join(MEDIA_DIR, mri_name) if mri_name else ""
        cam_path = os.path.join(MEDIA_DIR, cam_name) if cam_name else ""

        if seg_name and not os.path.exists(seg_path) and os.path.exists(mri_path) and os.path.exists(cam_path):
            try:
                orig_bgr = cv2.imread(mri_path)
                cam_bgr = cv2.imread(cam_path)
                if orig_bgr is not None and cam_bgr is not None:
                    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    cam_rgb = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB)
                    diff = np.clip(cam_rgb[:, :, 0].astype(float) - orig_rgb[:, :, 0].astype(float), 0, 255)
                    heat = diff / (np.max(diff) + 1e-6)
                    seg_res = generate_tumor_segmentation(heat, orig_rgb, result.get("predicted_class", ""))
                    cv2.imwrite(seg_path, cv2.cvtColor(seg_res["overlay"], cv2.COLOR_RGB2BGR))
                    result["seg_info"] = seg_res
            except Exception as e:
                logger.warning("Could not auto-generate segmented overlay: %s", e)

        result["seg_img_name"] = seg_name if (seg_name and os.path.exists(seg_path)) else None

        if "explanation" not in result or "score_explanation" not in result.get("explanation", {}):
            result["explanation"] = generate_diagnostic_explanation(
                pred_class=result.get("predicted_class", "Unknown"),
                confidence=result.get("confidence", 0.0),
                patient_info=result.get("patient_info", {}),
                probabilities=result.get("probabilities", {}),
            )

    return render_template(
        "analysis.html",
        active_page="analysis",
        result=result,
        all_analyses=all_analyses
    )


# ──────────────────────────────────────────────────────────────
#  Server Launch
# ──────────────────────────────────────────────────────────────

def main():
    initialise_database()
    init_models()
    port = 5000
    print(f"\n=======================================================")
    print(f" NeuroFusion AI")
    print(f" Menu Options: Home Page | New Intake | Patient Information")
    print(f" Local Web Server: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
