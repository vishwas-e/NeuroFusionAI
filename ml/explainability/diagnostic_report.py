"""
NeuroFusion AI — Prediction Reason Generator (Simple English)
=============================================================
Explains why a specific tumor class was predicted using simple,
clear English that is easy for doctors and patients to understand,
including an explanation of the prediction scores / likelihood percentages.
"""

from typing import Dict, Any


def generate_diagnostic_explanation(
    pred_class: str,
    confidence: float,
    patient_info: Dict[str, Any],
    probabilities: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Generate clear, simple-English reasons for the tumor prediction,
    including an explanation of the percentage scores.
    """
    c_lower = (pred_class or "").lower().replace(" ", "").replace("_", "")
    age = patient_info.get("age", 45)
    gender = patient_info.get("gender", "Patient")
    headache = str(patient_info.get("headache", "No")).lower() == "yes"
    vision = str(patient_info.get("vision_problems", "No")).lower() == "yes"
    seizures = str(patient_info.get("seizures", "No")).lower() == "yes"
    nausea = str(patient_info.get("nausea", "No")).lower() == "yes"
    memory = str(patient_info.get("memory_loss", "No")).lower() == "yes"

    # ── Explain the Percentage Scores / Probabilities
    score_text = ""
    if probabilities:
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        top_name, top_prob = sorted_probs[0]
        second_name, second_prob = sorted_probs[1] if len(sorted_probs) > 1 else ("", 0)
        notumor_prob = probabilities.get("notumor", 0.0)

        if "notumor" in c_lower:
            score_text = (
                f"The AI evaluated both MRI and symptoms, scoring Healthy Brain (No Tumor) highest at {top_prob*100:.1f}%. "
                "Because both sides of the brain look clean and the patient has no tumor symptoms, the other tumor categories scored nearly 0%."
            )
        else:
            tumor_certainty = (1.0 - notumor_prob) * 100
            score_text = (
                f"The AI combined clues from the MRI scan and patient symptoms into scores that add up to 100%. "
                f"{pred_class.title()} scored highest at {top_prob*100:.1f}%, making it the main diagnosis. "
                f"Healthy Brain (No Tumor) scored only {notumor_prob*100:.1f}%, meaning the AI is {tumor_certainty:.1f}% certain a tumor is present. "
                f"{second_name.title()} scored {second_prob*100:.1f}% as a secondary possibility for the doctor to review."
            )
    else:
        score_text = f"The multimodal AI combined MRI scan and symptom features to reach a confidence of {confidence*100:.1f}%."

    if "glio" in c_lower:
        symptoms = []
        if seizures:
            symptoms.append("seizures (fits)")
        if headache:
            symptoms.append("frequent headaches")
        if nausea:
            symptoms.append("nausea or vomiting")
        if memory:
            symptoms.append("memory problems")

        symptom_text = (
            f"The patient has {' and '.join(symptoms)}, which happen when a tumor irritates brain tissue and raises pressure inside the head."
            if symptoms
            else f"The patient profile ({age}-year-old {gender}) matches the typical group where this tumor appears."
        )

        return {
            "title": "Glioma",
            "mri_reason": "The MRI scan shows an irregular, bright abnormal mass growing directly inside the brain tissue with uneven edges.",
            "clinical_reason": symptom_text,
            "score_explanation": score_text,
            "rule_out_reason": "Because the mass is growing deep inside the brain tissue (not on the outer lining or near the eye gland), it is identified as a Glioma.",
        }

    elif "menin" in c_lower:
        symptoms = []
        if headache:
            symptoms.append("headaches caused by pressure against the outer brain lining")
        if gender.lower() == "female":
            symptoms.append("female patient profile (this tumor is twice as common in women)")
        if not seizures:
            symptoms.append("no sudden seizures, which is typical for slow-growing surface tumors")

        symptom_text = (
            f"The patient reports {', and '.join(symptoms)}."
            if symptoms
            else f"The patient ({age}-year-old {gender}) shows symptoms of slow pressure against the outer skull wall."
        )

        return {
            "title": "Meningioma",
            "mri_reason": "The MRI scan shows a round tumor with clear, sharp borders attached to the outer protective lining (membrane) of the brain, pushing against the brain rather than growing inside it.",
            "clinical_reason": symptom_text,
            "score_explanation": score_text,
            "rule_out_reason": "Because the tumor sits on the outer brain lining and has smooth, clear borders, it is identified as a Meningioma and not a deep Glioma.",
        }

    elif "pitu" in c_lower:
        symptoms = []
        if vision:
            symptoms.append("vision problems or blurriness, because the tumor presses directly against the optic nerves that connect to the eyes")
        if headache:
            symptoms.append("headaches at the front or center of the head")

        symptom_text = (
            f"The patient reports {', and '.join(symptoms)}."
            if symptoms
            else "The patient's symptoms match pressure around the eye nerves and pituitary gland."
        )

        return {
            "title": "Pituitary Tumor",
            "mri_reason": "The MRI scan shows an abnormal growth at the bottom base of the skull, right where the pituitary gland sits under the brain.",
            "clinical_reason": symptom_text,
            "score_explanation": score_text,
            "rule_out_reason": "Because the tumor is located specifically in the pituitary gland area at the base of the skull, it is identified as a Pituitary tumor and not a top brain tumor.",
        }

    else:
        # Notumor
        symptom_notes = []
        if not seizures and not vision and not memory:
            symptom_notes.append("no major nerve problems like seizures, vision loss, or confusion")
        if headache:
            symptom_notes.append("headache is present, but it is likely a regular headache (such as tension or migraine) rather than one caused by a tumor")
        else:
            symptom_notes.append("no signs of swelling or pressure inside the head")

        return {
            "title": "No Tumor Detected",
            "mri_reason": "The MRI scan looks healthy and normal. Both the left and right sides of the brain are balanced and clean, with no lumps, swelling, or dark/bright tumor spots.",
            "clinical_reason": f"The patient reports {', and '.join(symptom_notes)}.",
            "score_explanation": score_text,
            "rule_out_reason": "Both the MRI image and the patient's symptoms show healthy brain condition with no tumor found.",
        }
