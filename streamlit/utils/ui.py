from typing import Text

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


def set_page_container_style() -> None:
    margins_css = """
    <style>
        .main > div {
            max-width: 100%;
            padding-left:10%;
        }

        button[data-baseweb="tab"] div p {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    """

    st.markdown(margins_css, unsafe_allow_html=True)


def display_sidebar_header() -> None:
    logo = Image.open("/home/konradballegro/streamlit/static/logo.png")
    with st.sidebar:
        st.image(logo, use_column_width=True)
        col1, col2 = st.columns(2)
        repo_link: Text = "https://github.com/KonuTech/mlops-zoomcamp-project"
        evidently_docs: Text = "https://docs.evidentlyai.com"
        col1.markdown(
            f"<a style = 'display: block; text-align: center;' href={repo_link} > Source code </a>",
            unsafe_allow_html=True,
        )
        col2.markdown(
            f"<a style = 'display: block; text-align: center;' href={evidently_docs} > Documentation </a>",
            unsafe_allow_html=True,
        )
        st.header("")


def display_header(report_name: Text, window_size: int) -> None:
    st.header(f"Report name: {report_name}")
    st.caption(f"Window size: {window_size}")


def display_report(report: Text) -> Text:
    components.html(report, width=1500, height=1000, scrolling=True)
    return report
