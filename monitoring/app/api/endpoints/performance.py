"""Endpoint para cálculo de Performance."""
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

router = APIRouter(prefix="/performance")

_MODEL_PATH = Path(__file__).parents[3] / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _model = pickle.load(_f)


class Record(BaseModel):
    REF_DATE: str
    TARGET: int

    class Config:
        extra = "allow"


class PerformanceRequest(BaseModel):
    records: List[Record]


@router.post("")
def compute_performance(payload: PerformanceRequest):
    df = pd.DataFrame([r.dict() for r in payload.records])
    df = df.replace({None: np.nan})

    ref_dates = pd.to_datetime(df["REF_DATE"])
    volumetria = (
        df.groupby(ref_dates.dt.to_period("M").astype(str))
        .size()
        .to_dict()
    )

    X = df.drop(columns=["REF_DATE", "TARGET"])
    y = df["TARGET"]
    probas = _model.predict_proba(X)[:, 1]
    auc_roc = float(roc_auc_score(y, probas))

    return {"volumetria": volumetria, "auc_roc": auc_roc}
