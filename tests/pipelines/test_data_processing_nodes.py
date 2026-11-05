"""Testy nodów przygotowania danych."""

import pandas as pd

from asi_project.pipelines.data_processing.nodes import normalize_raw_data


def test_normalize_raw_data_cleans_headers_and_text() -> None:
    data = pd.DataFrame({"Call  Failure": [" 1 "], "Status": [" Active "]})

    normalized = normalize_raw_data(data)

    assert list(normalized.columns) == ["Call Failure", "Status"]
    assert normalized.loc[0, "Status"] == "Active"
