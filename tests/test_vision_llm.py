"""Unit tests for Phase 9.5 Nemotron VL multimodal reasoning client."""

from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.vision_llm import (
    NemotronVisionClient,
    VisionLLMError,
    VisionLLMObservation,
    VisionLLMResult,
    reason_over_media,
)


@pytest.fixture
def mock_settings(monkeypatch):
    """Provide a mock Settings instance with vision config."""
    settings = MagicMock()
    settings.aipipe_api_key = "test-api-key"
    settings.vision_model_name = "nvidia/nemotron-nano-12b-v2-vl:free"
    settings.vision_model_endpoint = "https://test.endpoint/v1"
    settings.vision_request_timeout_seconds = 30.0
    settings.vision_max_frames = 24
    settings.vision_temperature = 0.1
    settings.vision_max_output_tokens = 1024

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def mock_image(tmp_path: Path) -> Path:
    """Create a tiny test image."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color="red")
    path = tmp_path / "test.png"
    img.save(path)
    return path


@pytest.fixture
def mock_http_client():
    """Provide a mock httpx AsyncClient."""
    client = AsyncMock()
    return client


@pytest.fixture
def sample_api_response() -> Dict[str, Any]:
    """Sample successful Nemotron API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The image contains a red square with white text.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 15,
            "total_tokens": 135,
        },
    }


def test_vision_llm_observation_to_dict():
    """Test VisionLLMObservation serialization."""
    obs = VisionLLMObservation(
        role="assistant",
        content="Test content",
        raw={"extra": "metadata"},
    )
    result = obs.to_dict()
    assert result["role"] == "assistant"
    assert result["content"] == "Test content"
    assert result["raw"]["extra"] == "metadata"


def test_vision_llm_result_to_dict():
    """Test VisionLLMResult serialization."""
    obs = VisionLLMObservation(role="assistant", content="Answer", raw={})
    result = VisionLLMResult(
        answer="Answer text",
        observations=[obs],
        model="test-model",
        usage={"total_tokens": 100},
        metadata={"test": "value"},
    )
    payload = result.to_dict()
    assert payload["answer"] == "Answer text"
    assert len(payload["observations"]) == 1
    assert payload["model"] == "test-model"
    assert payload["usage"]["total_tokens"] == 100
    assert payload["metadata"]["test"] == "value"


@pytest.mark.anyio
async def test_nemotron_client_requires_api_key(monkeypatch):
    """Test that NemotronVisionClient fails without API key."""
    settings = MagicMock()
    settings.aipipe_api_key = None

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm, "get_settings", lambda: settings)

    with pytest.raises(VisionLLMError, match="AIPIPE_API_KEY must be configured"):
        NemotronVisionClient()


@pytest.mark.anyio
async def test_nemotron_client_lifecycle(mock_settings, mock_http_client):
    """Test NemotronVisionClient context manager."""
    client = NemotronVisionClient(http_client=mock_http_client)
    assert client._client is mock_http_client
    assert not client._owns_client

    async with client:
        pass

    # Should not close externally provided client
    mock_http_client.aclose.assert_not_called()


@pytest.mark.anyio
async def test_nemotron_client_owns_client_when_none_provided(
    mock_settings, monkeypatch
):
    """Test that client is created and closed when not provided."""
    mock_async_client_class = MagicMock()
    mock_instance = AsyncMock()
    mock_async_client_class.return_value = mock_instance

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", mock_async_client_class)

    client = NemotronVisionClient()
    assert client._owns_client

    await client.close()
    mock_instance.aclose.assert_called_once()


@pytest.mark.anyio
async def test_nemotron_client_headers(mock_settings):
    """Test that client constructs correct authorization headers."""
    client = NemotronVisionClient()
    headers = client._headers()
    assert headers["Authorization"] == "Bearer test-api-key"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.anyio
async def test_nemotron_reason_missing_media(mock_settings, mock_http_client):
    """Test that reason() raises when no media paths are provided."""
    client = NemotronVisionClient(http_client=mock_http_client)

    with pytest.raises(VisionLLMError, match="No media assets were provided"):
        await client.reason(question="What is this?", media_paths=[])


@pytest.mark.anyio
async def test_nemotron_reason_missing_file(
    mock_settings, mock_http_client, tmp_path: Path
):
    """Test that reason() raises when media file doesn't exist."""
    client = NemotronVisionClient(http_client=mock_http_client)
    missing = tmp_path / "missing.png"

    with pytest.raises(VisionLLMError, match="Media file not found"):
        await client.reason(question="What is this?", media_paths=[missing])


@pytest.mark.anyio
async def test_nemotron_reason_success(
    mock_settings, mock_http_client, mock_image, sample_api_response, monkeypatch
):
    """Test successful Nemotron reasoning call."""
    mock_response = MagicMock()
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    # Mock image encoding
    def mock_encode(path):
        return "base64encodeddata"

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm.data_tools, "image_to_base64", mock_encode)

    client = NemotronVisionClient(http_client=mock_http_client)
    result = await client.reason(
        question="What is in this image?",
        media_paths=[mock_image],
    )

    assert isinstance(result, VisionLLMResult)
    assert result.answer == "The image contains a red square with white text."
    assert result.model == "nvidia/nemotron-nano-12b-v2-vl:free"
    assert len(result.observations) == 1
    assert result.observations[0].role == "assistant"
    assert result.usage["total_tokens"] == 135

    # Verify API call was made
    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    assert "chat/completions" in call_args[0][0]  # First positional arg is the URL
    payload = call_args[1]["json"]
    assert payload["model"] == "nvidia/nemotron-nano-12b-v2-vl:free"
    assert "messages" in payload
    assert payload["messages"][0]["role"] == "user"


@pytest.mark.anyio
async def test_nemotron_reason_with_hints_and_instructions(
    mock_settings, mock_http_client, mock_image, sample_api_response, monkeypatch
):
    """Test that hints and instructions are properly included in the prompt."""
    mock_response = MagicMock()
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    def mock_encode(path):
        return "base64data"

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm.data_tools, "image_to_base64", mock_encode)

    client = NemotronVisionClient(http_client=mock_http_client)
    await client.reason(
        question="Describe this chart",
        media_paths=[mock_image],
        instructions="Answer in JSON format",
        hints=["Hint 1: High entropy", "Hint 2: Multiple colors"],
    )

    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    user_message = payload["messages"][0]
    text_content = [item for item in user_message["content"] if item["type"] == "text"][
        0
    ]
    text = text_content["text"]

    assert "Describe this chart" in text
    assert "Constraints: Answer in JSON format" in text
    assert "Hint 1: High entropy" in text
    assert "Hint 2: Multiple colors" in text


@pytest.mark.anyio
async def test_nemotron_reason_frame_truncation(
    mock_settings, mock_http_client, tmp_path: Path, sample_api_response, monkeypatch
):
    """Test that excessive frames are sampled down to max_frames."""
    from PIL import Image

    # Create 50 images but limit is 24
    mock_settings.vision_max_frames = 24
    images = []
    for i in range(50):
        img = Image.new("RGB", (32, 32), color="blue")
        path = tmp_path / f"frame_{i:03d}.png"
        img.save(path)
        images.append(path)

    mock_response = MagicMock()
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    def mock_encode(path):
        return f"base64_{path.stem}"

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm.data_tools, "image_to_base64", mock_encode)

    client = NemotronVisionClient(http_client=mock_http_client)
    result = await client.reason(question="Analyze video", media_paths=images)

    # Verify frame count in metadata
    assert result.metadata["media_processed"] <= 24

    # Verify API payload includes at most 24 images
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    user_content = payload["messages"][0]["content"]
    image_items = [item for item in user_content if item["type"] == "image_url"]
    assert len(image_items) <= 24


@pytest.mark.anyio
async def test_nemotron_reason_custom_temperature(
    mock_settings, mock_http_client, mock_image, sample_api_response, monkeypatch
):
    """Test that custom temperature overrides setting default."""
    mock_response = MagicMock()
    mock_response.json.return_value = sample_api_response
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    def mock_encode(path):
        return "base64data"

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm.data_tools, "image_to_base64", mock_encode)

    client = NemotronVisionClient(http_client=mock_http_client)
    await client.reason(question="Test", media_paths=[mock_image], temperature=0.7)

    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    assert payload["temperature"] == 0.7


@pytest.mark.anyio
async def test_nemotron_reason_no_choices_raises(
    mock_settings, mock_http_client, mock_image, monkeypatch
):
    """Test that missing choices in response raises VisionLLMError."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "test", "choices": []}
    mock_response.raise_for_status = MagicMock()
    mock_http_client.post.return_value = mock_response

    def mock_encode(path):
        return "base64data"

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm.data_tools, "image_to_base64", mock_encode)

    client = NemotronVisionClient(http_client=mock_http_client)

    with pytest.raises(VisionLLMError, match="did not include any choices"):
        await client.reason(question="Test", media_paths=[mock_image])


@pytest.mark.anyio
async def test_nemotron_parse_response_list_content():
    """Test parsing response when content is a list of blocks."""
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Part 1 "},
                        {"type": "text", "text": "Part 2"},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 50},
        "model": "test-model",
    }

    result = NemotronVisionClient._parse_response(payload, media_count=1)
    assert result.answer == "Part 1  Part 2"  # Extra space from join
    assert result.model == "test-model"


@pytest.mark.anyio
async def test_reason_over_media_convenience_wrapper(
    mock_settings, mock_image, sample_api_response, monkeypatch
):
    """Test the convenience wrapper creates and closes client automatically."""
    mock_client_instance = AsyncMock()
    mock_client_instance.reason.return_value = VisionLLMResult(
        answer="Test answer",
        observations=[],
        model="test-model",
    )
    mock_client_instance.close = AsyncMock()

    mock_client_class = MagicMock(return_value=mock_client_instance)

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm, "NemotronVisionClient", mock_client_class)

    result = await reason_over_media(
        question="What is this?",
        media_paths=[mock_image],
        hints=["Test hint"],
    )

    assert result.answer == "Test answer"
    mock_client_instance.reason.assert_called_once()
    mock_client_instance.close.assert_called_once()


@pytest.mark.anyio
async def test_reason_over_media_reuses_provided_client(
    mock_settings, mock_image, monkeypatch
):
    """Test that provided client is reused and not closed."""
    mock_client = AsyncMock()
    mock_client.reason.return_value = VisionLLMResult(
        answer="Reused client",
        observations=[],
    )
    mock_client.close = AsyncMock()

    result = await reason_over_media(
        question="Test",
        media_paths=[mock_image],
        client=mock_client,
    )

    assert result.answer == "Reused client"
    mock_client.reason.assert_called_once()
    # Should NOT close externally provided client
    mock_client.close.assert_not_called()


@pytest.mark.anyio
async def test_reason_over_media_forwards_all_parameters(
    mock_settings, mock_image, monkeypatch
):
    """Test that all parameters are forwarded correctly."""
    mock_client_instance = AsyncMock()
    mock_client_instance.reason.return_value = VisionLLMResult(
        answer="Answer",
        observations=[],
    )
    mock_client_instance.close = AsyncMock()

    mock_client_class = MagicMock(return_value=mock_client_instance)

    import app.vision_llm

    monkeypatch.setattr(app.vision_llm, "NemotronVisionClient", mock_client_class)

    await reason_over_media(
        question="Custom question",
        media_paths=[mock_image, mock_image],
        instructions="Return JSON",
        hints=["Hint A", "Hint B"],
        temperature=0.5,
        max_output_tokens=2048,
    )

    call_kwargs = mock_client_instance.reason.call_args[1]
    assert call_kwargs["question"] == "Custom question"
    assert len(call_kwargs["media_paths"]) == 2
    assert call_kwargs["instructions"] == "Return JSON"
    assert call_kwargs["hints"] == ["Hint A", "Hint B"]
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_output_tokens"] == 2048
