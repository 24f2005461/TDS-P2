import base64
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from app import data_tools


def test_load_dataframe_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("col1,col2\n1,foo\n2,bar\n", encoding="utf-8")

    df = data_tools.load_dataframe(csv_path)

    assert list(df.columns) == ["col1", "col2"]
    assert df.shape == (2, 2)
    assert df.iloc[0]["col2"] == "foo"


def test_summarize_dataframe_returns_expected_metadata() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    summary = data_tools.summarize_dataframe(df)

    assert summary.row_count == 3
    assert summary.column_count == 2
    assert summary.columns == ["x", "y"]


def test_read_json_payload(tmp_path: Path) -> None:
    payload = {"answer": 42, "notes": ["foo", "bar"]}
    json_path = tmp_path / "payload.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    result = data_tools.read_json_payload(json_path)

    assert result == payload


def test_read_text_file_with_whitespace_deduplication(tmp_path: Path) -> None:
    text_path = tmp_path / "notes.txt"
    text_path.write_text("Line 1\n\nLine    2   here", encoding="utf-8")

    deduped = data_tools.read_text_file(text_path, dedupe_whitespace=True)

    assert deduped == "Line 1 Line 2 here"


def test_load_image_and_image_to_base64(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    image = Image.new("RGB", (10, 10), color="red")
    image.save(image_path, format="PNG")

    loaded = data_tools.load_image(image_path)
    assert loaded.size == (10, 10)
    assert loaded.mode == "RGB"

    encoded = data_tools.image_to_base64(image_path)
    decoded_bytes = base64.b64decode(encoded)
    assert decoded_bytes.startswith(b"\x89PNG")

    roundtrip_image = Image.open(BytesIO(decoded_bytes))
    assert roundtrip_image.size == (10, 10)


def test_extract_pdf_text_single_page(tmp_path: Path) -> None:
    # Create a minimal PDF using reportlab-like approach via PIL (PNG -> PDF)
    image = Image.new("RGB", (100, 50), color="white")
    pdf_path = tmp_path / "doc.pdf"
    image.save(pdf_path, "PDF")

    text = data_tools.extract_pdf_text(pdf_path, max_pages=1)
    # Generated PDF will be empty text, ensure call succeeds
    assert isinstance(text, str)


def test_dataframe_to_records_and_chunk_dataframe() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    records = data_tools.dataframe_to_records(df)
    assert records[0] == {"a": 1, "b": "w"}
    assert len(records) == 4

    chunks = list(data_tools.chunk_dataframe(df, chunk_size=2))
    assert len(chunks) == 2
    assert chunks[0].shape == (2, 2)
    assert chunks[1].iloc[-1]["b"] == "z"
