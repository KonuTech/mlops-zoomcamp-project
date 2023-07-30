"""
Evidently Utility Functions

This script contains utility functions
to work with Evidently library for model performance monitoring.

Functions:
    - get_column_mapping: Create a ColumnMapping object based on specified arguments.
    - build_model_performance_report: Build a model performance report using Evidently.
    - build_target_drift_report: Build a target drift report using Evidently.
"""

from typing import Text

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import (
    RegressionAbsPercentageErrorPlot,
    RegressionErrorDistribution,
    RegressionErrorNormality,
    RegressionErrorPlot,
    RegressionPredictedVsActualPlot,
    RegressionPredictedVsActualScatter,
    RegressionQualityMetric,
    RegressionTopErrorMetric,
)
from evidently.report import Report


def get_column_mapping(**kwargs) -> ColumnMapping:
    """
    Create a ColumnMapping object based on specified arguments.

    Args:
        **kwargs: Keyword arguments containing 'target_col', 'prediction_col', and 'num_features'.

    Returns:
        ColumnMapping: A ColumnMapping object containing column mapping information.
    """
    column_mapping = ColumnMapping()
    column_mapping.target = kwargs["target_col"]
    column_mapping.prediction = kwargs["prediction_col"]
    column_mapping.numerical_features = kwargs["num_features"]

    return column_mapping


def build_model_performance_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> Text:
    """
    Build a model performance report using Evidently.

    Args:
        reference_data (pd.DataFrame): DataFrame containing the reference data.
        current_data (pd.DataFrame): DataFrame containing the current data.
        column_mapping (ColumnMapping): ColumnMapping object containing column mapping information.

    Returns:
        Text: File path of the generated model performance report in HTML format.
    """
    model_performance_report = Report(
        metrics=[
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(),
            RegressionPredictedVsActualPlot(),
            RegressionErrorPlot(),
            RegressionAbsPercentageErrorPlot(),
            RegressionErrorDistribution(),
            RegressionErrorNormality(),
            RegressionTopErrorMetric(),
        ]
    )
    model_performance_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    report_path = "reports/model_performance.html"
    model_performance_report.save_html(report_path)

    return report_path


def build_target_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> Text:
    """
    Build a target drift report using Evidently.

    Args:
        reference_data (pd.DataFrame): DataFrame containing the reference data.
        current_data (pd.DataFrame): DataFrame containing the current data.
        column_mapping (ColumnMapping): ColumnMapping object
        containing column mapping information.
    Returns:
        Text: File path of the generated target drift report in HTML format.
    """
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    report_path = "reports/target_drift.html"
    target_drift_report.save_html(report_path)

    return report_path
