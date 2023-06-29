import pandas as pd
from fastapi import FastAPI
from fastapi.responses import (HTMLResponse, FileResponse)

from src.utils.data import load_current_data


app = FastAPI()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse("<h1><i>Evidently + FastAPI</i></h1>")


def monitor_model_performance(window_size: int = 1000) -> FileResponse:
    current_data: pd.DataFrame = load_current_data()