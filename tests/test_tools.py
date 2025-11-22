from pathlib import Path

import pytest

import app.tools as tools


@pytest.mark.anyio
async def test_register_tool_adds_callable():
    async def sample_tool():
        return "ok"

    tools.registry.register("sample_tool", sample_tool)
    assert tools.registry.get("sample_tool") is sample_tool


@pytest.mark.anyio
async def test_download_resource_delegates_to_downloader(monkeypatch):
    sentinel_result = object()

    async def fake_download(url, destination_dir):
        assert url == "https://example.com/data.csv"
        assert destination_dir == Path("/tmp")
        return sentinel_result

    monkeypatch.setattr(tools, "download_file", fake_download)
    result = await tools.download_resource("https://example.com/data.csv", Path("/tmp"))
    assert result is sentinel_result


@pytest.mark.anyio
async def test_load_table_uses_data_tools(monkeypatch):
    df = object()

    def fake_load(path, file_type=None):
        assert path == Path("table.csv")
        assert file_type == "csv"
        return df

    monkeypatch.setattr(tools.data_tools, "load_dataframe", fake_load)
    result = await tools.load_table(Path("table.csv"), file_type="csv")
    assert result is df


@pytest.mark.anyio
async def test_run_python_executes_code():
    code = "print('hello'); value = 40 + 2"
    result = await tools.run_python(code)
    assert result.success is True
    assert "hello" in result.stdout
    assert result.stderr == ""


@pytest.mark.anyio
async def test_create_chart_generates_base64():
    data = [{"x": 0, "y": 0}, {"x": 1, "y": 2}]
    result = await tools.create_chart(data, x="x", y="y")
    assert result["chart_type"] == "line"
    assert isinstance(result["image_base64"], str)
    assert len(result["image_base64"]) > 10


def test_tool_registry_contains_default_bindings():
    default_tools = {
        "download_resource",
        "download_resources",
        "load_table",
        "save_table",
        "summarize_table",
        "describe_dataframe",
        "value_counts",
        "chunk_dataframe",
        "dataframe_to_records",
        "read_csv",
        "read_excel",
        "read_json",
        "read_text",
        "write_text",
        "extract_pdf_text",
        "extract_pdf_tables",
        "read_pdf",
        "load_image",
        "image_to_base64",
        "encode_image",
        "create_chart",
        "analyze_image",
        "vision_reasoner",
        "reason_over_image",
        "call_json_api",
        "parse_instruction_headers",
        "parse_instruction_text",
        "run_python",
        "run_trusted",
    }
    assert default_tools.issubset(set(tools.registry.names()))
