"""
NeuroFusion AI — Clinical Data Preprocessing
=============================================
Handles encoding, scaling, missing-value imputation, and
conversion of patient clinical features into numeric vectors.

Algorithms: Categorical Encoding, Feature Scaling
"""

import os, sys, logging, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  ClinicalPreprocessor
# ──────────────────────────────────────────────────────────────

class ClinicalPreprocessor:
    """
    Fit-transform pipeline for clinical/patient features.

    Workflow
    --------
    1. Fill missing numerical values with the median.
    2. Fill missing categorical values with the mode.
    3. Ordinal-encode each categorical feature with LabelEncoder.
    4. Standardise all features with StandardScaler.
    5. Return a 1-D float32 numpy array ready for the neural network.

    The fitted encoders and scaler are saved to disk so the same
    transformation can be applied at inference time without re-fitting.
    """

    def __init__(self,
                 numerical_features   = None,
                 categorical_features = None):
        self.numerical_features   = numerical_features   or config.CLINICAL_NUMERICAL_FEATURES
        self.categorical_features = categorical_features or config.CLINICAL_CATEGORICAL_FEATURES
        self.all_features         = self.numerical_features + self.categorical_features

        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self._is_fitted = False

        # Store medians/modes for imputation at inference
        self._num_medians: dict = {}
        self._cat_modes  : dict = {}

    # ── Fit ──────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ClinicalPreprocessor":
        """
        Fit encoders and scaler on a training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the columns listed in
            self.all_features.  Extra columns are ignored.
        """
        df = df.copy()

        # Imputation stats
        for col in self.numerical_features:
            if col in df.columns:
                self._num_medians[col] = df[col].median()

        for col in self.categorical_features:
            if col in df.columns:
                self._cat_modes[col] = df[col].mode().iloc[0]

        # Fill missing
        df = self._impute(df)

        # Label-encode categoricals
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Fit scaler on all features
        present = [c for c in self.all_features if c in df.columns]
        self.scaler.fit(df[present].values.astype(np.float32))
        self._is_fitted = True
        logger.info("ClinicalPreprocessor fitted on %d samples.", len(df))
        return self

    # ── Transform ────────────────────────────────────────────

    def transform(self, data) -> np.ndarray:
        """
        Transform clinical data to a normalised float32 vector.

        Parameters
        ----------
        data : pd.DataFrame | dict
            A single row (dict) or a DataFrame.

        Returns
        -------
        np.ndarray  shape (N, n_features) or (n_features,)
        """
        if not self._is_fitted:
            raise RuntimeError("ClinicalPreprocessor is not fitted. Call fit() first.")

        if isinstance(data, dict):
            df = pd.DataFrame([data])
            single = True
        else:
            df = data.copy()
            single = False

        df = self._impute(df)

        # Encode categoricals
        for col in self.categorical_features:
            if col in df.columns:
                le = self.label_encoders.get(col)
                if le is not None:
                    df[col] = df[col].astype(str).apply(
                        lambda v: le.transform([v])[0]
                        if v in le.classes_
                        else 0                        # unseen label → 0
                    )

        present = [c for c in self.all_features if c in df.columns]
        X = self.scaler.transform(df[present].values.astype(np.float32))

        return X[0] if single else X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convenience: fit then transform the same DataFrame."""
        return self.fit(df).transform(df)

    # ── Persist ──────────────────────────────────────────────

    def save(self, scaler_path: str = config.SCALER_PATH,
             encoder_path: str = config.LABEL_ENCODER_PATH):
        """Serialise the fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(scaler_path),  exist_ok=True)
        os.makedirs(os.path.dirname(encoder_path), exist_ok=True)

        with open(scaler_path,  "wb") as f:
            pickle.dump(self.scaler, f)
        with open(encoder_path, "wb") as f:
            pickle.dump({
                "label_encoders"    : self.label_encoders,
                "num_medians"       : self._num_medians,
                "cat_modes"         : self._cat_modes,
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
            }, f)
        logger.info("ClinicalPreprocessor saved.")

    @classmethod
    def load(cls,
             scaler_path: str = config.SCALER_PATH,
             encoder_path: str = config.LABEL_ENCODER_PATH) -> "ClinicalPreprocessor":
        """Load a previously saved preprocessor."""
        with open(scaler_path,  "rb") as f:
            scaler = pickle.load(f)
        with open(encoder_path, "rb") as f:
            state  = pickle.load(f)

        obj = cls(
            numerical_features=state["numerical_features"],
            categorical_features=state["categorical_features"],
        )
        obj.scaler          = scaler
        obj.label_encoders  = state["label_encoders"]
        obj._num_medians    = state["num_medians"]
        obj._cat_modes      = state["cat_modes"]
        obj._is_fitted      = True
        logger.info("ClinicalPreprocessor loaded from disk.")
        return obj

    # ── Private helpers ───────────────────────────────────────

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using stored medians / modes."""
        for col, med in self._num_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(med)
        for col, mode in self._cat_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode)
        return df

    @property
    def feature_dim(self) -> int:
        """Number of features after preprocessing (for model input)."""
        return len(self.all_features)


# ──────────────────────────────────────────────────────────────
#  Standalone helpers
# ──────────────────────────────────────────────────────────────

def build_clinical_dataframe(patient_info: dict) -> pd.DataFrame:
    """
    Convert a patient-info dict (from the Streamlit form) into a
    single-row DataFrame aligned to config.CLINICAL_ALL_FEATURES.

    Missing keys are filled with NaN for imputation.
    """
    row = {col: patient_info.get(col, np.nan)
           for col in config.CLINICAL_ALL_FEATURES}
    return pd.DataFrame([row])


def generate_synthetic_clinical_data(n_samples: int = 500,
                                     class_names: list = None,
                                     seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic clinical data for unit testing or demo mode.

    NOTE: This data is entirely artificial and must NOT be used for
    real clinical analysis or to train a production model.

    Parameters
    ----------
    n_samples : int
    class_names : list[str], optional
    seed : int

    Returns
    -------
    pd.DataFrame  with clinical columns + 'tumor_class' label
    """
    rng = np.random.default_rng(seed)
    if class_names is None:
        class_names = config.DEFAULT_CLASSES

    n = n_samples
    df = pd.DataFrame({
        "patient_id"      : [f"P{i:04d}" for i in range(n)],
        "age"             : rng.integers(10, 85, size=n),
        "gender"          : rng.choice(["Male", "Female"], size=n),
        "headache"        : rng.choice(["No", "Yes"], size=n, p=[0.4, 0.6]),
        "vision_problems" : rng.choice(["No", "Yes"], size=n, p=[0.6, 0.4]),
        "seizures"        : rng.choice(["No", "Yes"], size=n, p=[0.7, 0.3]),
        "nausea"          : rng.choice(["No", "Yes"], size=n, p=[0.5, 0.5]),
        "memory_loss"     : rng.choice(["No", "Yes"], size=n, p=[0.65, 0.35]),
        "tumor_class"     : rng.choice(class_names, size=n),
    })
    logger.info("Generated %d synthetic clinical samples (DEMO ONLY).", n)
    return df
