"""
Evidently + FastAPI Integration

This script defines a FastAPI application that provides endpoints for monitoring model performance
and target drift using the Evidently library. The application reads reference and current data,
generates reports, and returns the reports as FileResponse.

Endpoints:
    - '/' - A simple HTML response.
    - '/monitor-model' - Endpoint to monitor model performance and generate a report.
    - '/monitor-target' - Endpoint to monitor target drift and generate a report.
"""

from typing import Text

from config.config import DATA_COLUMNS
from evidently import ColumnMapping
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from src.utils.data import load_current_data, load_reference_data
from src.utils.reports import (
    build_model_performance_report,
    build_target_drift_report,
    get_column_mapping,
)

app = FastAPI()


@app.get("/")
def index() -> HTMLResponse:
    """
    A FastAPI endpoint to display a simple HTML response.

    Returns:
        HTMLResponse: A simple HTML response with a header.
    """
    return HTMLResponse("<h1><i>Evidently + FastAPI</i></h1>")


@app.get("/monitor-model")
def monitor_model_performance() -> FileResponse:
    """
    A FastAPI endpoint to monitor model performance and generate a report.

    Returns:
        FileResponse: The generated report file as a FileResponse.
    """
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
def monitor_target_drift() -> FileResponse:
    """
    A FastAPI endpoint to monitor target drift and generate a report.

    Returns:
        FileResponse: The generated report file as a FileResponse.
    """
    reference_data = load_reference_data(columns=DATA_COLUMNS["columns"])
    current_data = load_current_data(columns=DATA_COLUMNS["columns"])

    column_mapping: ColumnMapping = get_column_mapping(**DATA_COLUMNS)
    report_path: Text = build_target_drift_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    return FileResponse(report_path)
