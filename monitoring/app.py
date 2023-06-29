from typing import Text

import pandas as pd
from evidently import ColumnMapping
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from config.config import DATA_COLUMNS
from src.utils.data import load_current_data, load_reference_data
from src.utils.reports import (
    build_model_performance_report,
    build_target_drift_report,
    get_column_mapping,
)

app = FastAPI()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse("<h1><i>Evidently + FastAPI</i></h1>")


@app.get("/monitor-model")
def monitor_model_performance(window_size: int = 1000) -> FileResponse:
    reference_data = load_reference_data(columns=DATA_COLUMNS["columns"])
    current_data = load_current_data(columns=DATA_COLUMNS["columns"])

    column_mapping: ColumnMapping = get_column_mapping(**DATA_COLUMNS)
    report_path: Text = build_model_performance_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    return FileResponse(report_path)


@app.get("/monitor-target")
def monitor_target_drift(window_size: int = 1000) -> FileResponse:
    reference_data = load_reference_data(columns=DATA_COLUMNS["columns"])
    current_data = load_current_data(columns=DATA_COLUMNS["columns"])

    column_mapping: ColumnMapping = get_column_mapping(**DATA_COLUMNS)
    report_path: Text = build_target_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    return FileResponse(report_path)
