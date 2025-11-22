import base64
from pathlib import Path

import pandas as pd
import pytest

from app.visualization import (
    ChartArtifact,
    create_chart_from_records,
    render_matplotlib_chart,
    render_plotly_chart,
)


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2021, 2022, 2023],
            "value": [10, 15, 20],
        }
    )


def test_create_chart_from_records_returns_chart_metadata():
    data = [{"year": 2021, "value": 5}, {"year": 2022, "value": 9}]
    artifact = create_chart_from_records(
        data,
        x="year",
        y="value",
        chart_type="bar",
        engine="matplotlib",
        title="Demo",
        extra_metadata={"source": "unit-test"},
    )

    assert artifact.metadata["chart_type"] == "bar"
    assert isinstance(artifact.image_base64, str) and len(artifact.image_base64) > 10
    assert artifact.metadata["library"] == "matplotlib"
    assert artifact.metadata["source"] == "unit-test"


def test_render_matplotlib_chart_produces_decodable_png():
    df = _sample_dataframe()
    artifact = render_matplotlib_chart(
        df, x="year", y="value", chart_type="line", title="Growth"
    )

    raw = base64.b64decode(artifact.image_base64)
    assert raw.startswith(b"\x89PNG")
    assert artifact.metadata["chart_type"] == "line"
    assert artifact.metadata["library"] == "matplotlib"


@pytest.mark.anyio
async def test_render_plotly_chart_writes_file(tmp_path: Path):
    df = _sample_dataframe()
    output_path = tmp_path / "plot.png"
    artifact = render_plotly_chart(
        df,
        x="year",
        y="value",
        chart_type="scatter",
        title="Trend",
        save_path=output_path,
    )

    assert output_path.exists()
    assert artifact.path == output_path
    assert artifact.metadata["library"] == "plotly"


def test_chart_artifact_to_dict_handles_none_path():
    artifact = ChartArtifact(
        image_base64="Zm9vYmFy", path=None, metadata={"foo": "bar"}
    )
    payload = artifact.to_dict()

    assert payload["image_base64"] == "Zm9vYmFy"
    assert payload["path"] is None
    assert payload["metadata"] == {"foo": "bar"}
