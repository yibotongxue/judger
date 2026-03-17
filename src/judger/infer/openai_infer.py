from __future__ import annotations

from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI

from judger.infer.infer_protocol import InferProtocol
from judger.types import InferParameters, InferResult, MessageDataItem

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class OpenAIInfer(InferProtocol):
    """OpenAI inference implementation.

    This class uses the OpenAI SDK to perform inference via the chat completions API.
    The model and base_url must be explicitly provided by the user.

    Example:
        ```python
        inferer = OpenAIInfer(
            model="gpt-4o",
            api_key="your-api-key",
            base_url="https://api.openai.com/v1",
        )
        result = await inferer.infer(messages, parameters)
        ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI inference client.

        Args:
            model: The model ID to use (e.g., "gpt-4o", "gpt-3.5-turbo").
                Must be explicitly provided by the user.
            api_key: The API key for authentication.
            base_url: The base URL for the OpenAI API. Must be explicitly
                provided by the user if needed (e.g., for custom endpoints).

        Raises:
            ValueError: If model is empty or not provided.
        """
        if not model:
            raise ValueError("model must be explicitly provided")

        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    async def infer(
        self,
        messages: list[MessageDataItem],
        parameters: InferParameters,
    ) -> InferResult:
        """Perform inference using OpenAI API.

        Args:
            messages: List of messages for the conversation.
            parameters: Inference parameters (temperature, top_p, etc.).

        Returns:
            InferResult containing the response and metadata.
        """
        # Convert MessageDataItem list to OpenAI message format
        api_messages: list[ChatCompletionMessageParam] = [
            cast(ChatCompletionMessageParam, {"role": msg.role, "content": msg.content})
            for msg in messages
        ]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=api_messages,
            temperature=parameters.temperature,
            top_p=parameters.top_p,
        )

        assistant_message = response.choices[0].message.content or ""

        # Build next_messages by appending the assistant response
        next_messages = messages + [
            MessageDataItem(role="assistant", content=assistant_message)
        ]

        return InferResult(
            response=assistant_message,
            next_messages=next_messages,
            meta={
                "model_id": self._model,
                "infer_parameters": parameters,
            },
        )
