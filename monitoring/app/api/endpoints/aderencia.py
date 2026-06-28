"""Endpoint para cálculo de aderência."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

router = APIRouter(prefix="/aderencia")

_MODEL_PATH = Path(__file__).parents[3] / "model.pkl"
_REFERENCE_PATH = Path(__file__).parents[4] / "datasets/credit_01/test.gz"

_METADATA_COLS = {"REF_DATE", "TARGET", "ID"}

with open(_MODEL_PATH, "rb") as _f:
    _model = pickle.load(_f)


def _patch_ohe_handle_unknown(estimator) -> None:
    """Walk a fitted pipeline and set handle_unknown='ignore' on every OneHotEncoder.

    The model was trained without handle_unknown='ignore', so OOT data with new
    category values raises ValueError. Setting 'ignore' makes the OHE emit an
    all-zero row for unseen categories instead of crashing — semantically equivalent
    to "none of the known categories matched", which is the safest default without
    retraining.
    """
    if isinstance(estimator, OneHotEncoder):
        estimator.handle_unknown = "ignore"
    elif isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _patch_ohe_handle_unknown(step)
    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers_:
            if not isinstance(transformer, str):  # skip 'drop' / 'passthrough'
                _patch_ohe_handle_unknown(transformer)


_patch_ohe_handle_unknown(_model)


def _score(df: pd.DataFrame) -> np.ndarray:
    X = df.drop(columns=[c for c in _METADATA_COLS if c in df.columns])
    X = X.replace({None: np.nan})
    return _model.predict_proba(X)[:, 1]


_ref_scores = _score(pd.read_csv(_REFERENCE_PATH))


class AderenciaRequest(BaseModel):
    dataset_path: str


@router.post("")
def compute_aderencia(payload: AderenciaRequest):
    path = Path(payload.dataset_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {payload.dataset_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to read dataset: {exc}")

    input_scores = _score(df)
    ks_stat, p_value = ks_2samp(_ref_scores, input_scores)

    return {"ks_statistic": float(ks_stat), "p_value": float(p_value)}
