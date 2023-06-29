import os
from typing import Text

import requests
import streamlit as st

from utils.ui import (
    display_header,
    display_report,
    display_sidebar_header,
    set_page_container_style,
)

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

        report_selected: bool = False
        request_url: Text = base_route
        report_name: Text = ""
        if clicked_model_performance:
            report_selected = True
            request_url += f"/monior-model?window_size={window_size}"
            report_name = "Target performance"
        if clicked_target_drift:
            report_selected = True
            request_url += f"/monior-target?window_size={window_size}"
            report_name = "Target drift"
        if report_selected:
            resp: requests.Response = requests.get(request_url)
            display_header(report_name, window_size)
            display_report(resp.content)
        print("hello world")
    except Exception as e:
        st.error(e)
