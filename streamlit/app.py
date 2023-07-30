"""
This script provides a Streamlit web application
to display model performance and target drift reports
using data from a FastAPI server.
"""

import os
from typing import Text

import requests
from requests.exceptions import HTTPError, RequestException
from utils.ui import (
    display_header,
    display_report,
    display_sidebar_header,
    set_page_container_style,
)

import streamlit as st

if __name__ == "__main__":
    set_page_container_style()
    display_sidebar_header()
    host: Text = os.getenv("FASTAPI_API_HOST", "localhost")
    base_route: Text = f"http://{host}:8000"
    try:
        window_size: int = st.sidebar.number_input(
            label="window_size", min_value=1, step=1, value=1000
        )
        clicked_model_performance: bool = st.sidebar.button(label="Model performance")
        clicked_target_drift: bool = st.sidebar.button(label="Target drift")

        REPORT_SELECTED: bool = False
        request_url: Text = base_route
        REPORT_NAME: Text = ""
        if clicked_model_performance:
            REPORT_SELECTED = True
            request_url += f"/monitor-model?window_size={window_size}"
            REPORT_NAME = "Target performance"
        if clicked_target_drift:
            REPORT_SELECTED = True
            request_url += f"/monitor-target?window_size={window_size}"
            REPORT_NAME = "Target drift"
        if REPORT_SELECTED:
            resp: requests.Response = requests.get(request_url, timeout=10)
            display_header(REPORT_NAME, window_size)
            display_report(resp.content)
    except HTTPError as http_ex:
        st.error(f"HTTP Error: {http_ex}")
    except RequestException as req_ex:
        st.error(f"Request Exception: {req_ex}")
