"""
Streamlit Utility Functions

This script contains utility functions
for working with Streamlit to display reports and set page styles.

Functions:
    - set_page_container_style: Set the page container style for Streamlit app.
    - display_sidebar_header: Display the sidebar header with logo and links.
    - display_header: Display the header of the report with report name and window size.
    - display_report: Display the HTML report using Streamlit components.
"""

from typing import Text

from PIL import Image

import streamlit as st
import streamlit.components.v1 as components


def set_page_container_style() -> None:
    """
    Set the page container style for the Streamlit app.

    This function sets custom CSS styles to modify the appearance of the page container.

    Returns:
        None
    """
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
    """
    Display the sidebar header with logo and links.

    This function displays the logo and two links in the Streamlit sidebar.

    Returns:
        None
    """
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
            f"<a style = 'display: block; text-align: center;' href={evidently_docs} >"
            " Documentation </a>",
            unsafe_allow_html=True,
        )
        st.header("")


def display_header(report_name: Text, window_size: int) -> None:
    """
    Display the header of the report with report name and window size.

    Args:
        report_name (Text): The name of the report.
        window_size (int): The window size of the report.

    Returns:
        None
    """
    st.header(f"Report name: {report_name}")
    st.caption(f"Window size: {window_size}")


def display_report(report: Text) -> Text:
    """
    Display the HTML report using Streamlit components.

    Args:
        report (Text): The HTML report to be displayed.

    Returns:
        Text: The HTML report.
    """
    components.html(report, width=1500, height=1000, scrolling=True)
    return report
