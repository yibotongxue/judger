"""Tests for judger.infer.openai_infer module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from judger.infer.infer_protocol import InferProtocol
from judger.infer.openai_infer import OpenAIInfer
from judger.types import InferParameters, MessageDataItem


class TestOpenAIInfer:
    """Tests for OpenAIInfer class."""

    def test_init_requires_model(self):
        """Test that model must be explicitly provided."""
        with pytest.raises(ValueError, match="model must be explicitly provided"):
            OpenAIInfer(
                model="",
                api_key="test-api-key",
            )

    def test_init_with_required_params(self):
        """Test initialization with required parameters."""
        inferer = OpenAIInfer(
            model="gpt-4o",
            api_key="test-api-key",
        )
        assert inferer._model == "gpt-4o"
        assert isinstance(inferer, InferProtocol)

    def test_init_with_base_url(self):
        """Test initialization with custom base_url."""
        inferer = OpenAIInfer(
            model="gpt-4o",
            api_key="test-api-key",
            base_url="https://custom.api.com/v1",
        )
        assert inferer._model == "gpt-4o"

    def test_implements_protocol(self):
        """Test that OpenAIInfer implements InferProtocol."""
        assert issubclass(OpenAIInfer, InferProtocol)


class TestOpenAIInferAsync:
    """Async tests for OpenAIInfer class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Fixture to mock AsyncOpenAI client."""
        with patch("judger.infer.openai_infer.AsyncOpenAI") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def inferer(self):
        """Fixture to create OpenAIInfer instance."""
        return OpenAIInfer(
            model="gpt-4o",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
        )

    async def test_infer_success(self, mock_openai_client, inferer):
        """Test successful inference call."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, this is a test response"
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        # Prepare test data
        messages = [
            MessageDataItem(role="system", content="You are a helpful assistant"),
            MessageDataItem(role="user", content="Hello"),
        ]
        params = InferParameters(temperature=0.7, top_p=0.9)

        # Call infer
        result = await inferer.infer(messages, params)

        # Verify the result
        assert result.response == "Hello, this is a test response"
        assert len(result.next_messages) == 3
        assert result.next_messages[0].role == "system"
        assert result.next_messages[1].role == "user"
        assert result.next_messages[2].role == "assistant"
        assert result.next_messages[2].content == "Hello, this is a test response"
        assert result.meta["model_id"] == "gpt-4o"
        assert result.meta["infer_parameters"] == params

        # Verify API call
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["top_p"] == 0.9
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "You are a helpful assistant"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "Hello"

    async def test_infer_empty_response(self, mock_openai_client, inferer):
        """Test handling of empty response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [MessageDataItem(role="user", content="Hello")]
        params = InferParameters()

        result = await inferer.infer(messages, params)

        assert result.response == ""
        assert len(result.next_messages) == 2
        assert result.next_messages[1].content == ""

    async def test_infer_with_default_parameters(self, mock_openai_client, inferer):
        """Test inference with default parameters."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [MessageDataItem(role="user", content="Hello")]
        params = InferParameters()  # Uses defaults

        await inferer.infer(messages, params)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 1.0
        assert call_kwargs["top_p"] == 0.95

    async def test_infer_preserves_message_history(self, mock_openai_client, inferer):
        """Test that inference preserves and extends message history correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        original_messages = [
            MessageDataItem(role="system", content="System prompt"),
            MessageDataItem(role="user", content="Question 1"),
            MessageDataItem(role="assistant", content="Answer 1"),
            MessageDataItem(role="user", content="Question 2"),
        ]
        params = InferParameters()

        result = await inferer.infer(original_messages, params)

        # Original messages should not be modified
        assert len(original_messages) == 4

        # next_messages should have the assistant response appended
        assert len(result.next_messages) == 5
        assert result.next_messages[4].role == "assistant"
        assert result.next_messages[4].content == "Response"


class TestOpenAIInferDifferentModels:
    """Tests for OpenAIInfer with different model configurations."""

    def test_init_with_gpt35(self):
        """Test initialization with GPT-3.5 model."""
        inferer = OpenAIInfer(
            model="gpt-3.5-turbo",
            api_key="test-api-key",
        )
        assert inferer._model == "gpt-3.5-turbo"

    def test_init_with_custom_model(self):
        """Test initialization with custom model endpoint."""
        inferer = OpenAIInfer(
            model="custom-model-v1",
            api_key="test-api-key",
            base_url="https://custom-api.example.com/v1",
        )
        assert inferer._model == "custom-model-v1"
