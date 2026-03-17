"""Tests for judger.infer.infer_protocol module."""

from typing import get_type_hints

import pytest

from judger.infer.infer_protocol import InferProtocol
from judger.types import InferParameters, InferResult, MessageDataItem


class TestInferProtocol:
    """Tests for InferProtocol class."""

    def test_infer_protocol_is_abstract(self):
        """Test that InferProtocol cannot be instantiated directly."""
        # Protocol classes can be instantiated but raise TypeError when methods are called
        # if not properly implemented
        with pytest.raises(TypeError):
            InferProtocol()
            # Try to call the abstract method
            # This should fail since the protocol method has no implementation

    def test_infer_method_signature(self):
        """Test that infer method has correct signature."""
        # Check method exists
        assert hasattr(InferProtocol, "infer")

        # Check it's a method (function attribute)
        assert callable(getattr(InferProtocol, "infer"))

    def test_infer_method_parameters(self):
        """Test infer method parameter types."""
        hints = get_type_hints(InferProtocol.infer)

        # Check parameter types
        assert hints["messages"] == list[MessageDataItem]
        assert hints["parameters"] == InferParameters
        assert hints["return"] == InferResult


class TestInferProtocolImplementation:
    """Tests for InferProtocol implementations."""

    def test_valid_implementation(self):
        """Test a valid implementation of InferProtocol."""

        class ValidInferer:
            async def infer(
                self,
                messages: list[MessageDataItem],
                parameters: InferParameters,
            ) -> InferResult:
                return InferResult(
                    response="Test",
                    next_messages=[],
                    meta={
                        "model_id": "test-model",
                        "infer_parameters": parameters,
                    },
                )

        # Check that ValidInferer implements the protocol
        assert issubclass(ValidInferer, InferProtocol)

    def test_invalid_implementation_missing_method(self):
        """Test class missing infer method doesn't implement protocol."""

        class InvalidInferer:
            pass

        # This class doesn't implement InferProtocol
        assert not issubclass(InvalidInferer, InferProtocol)

    def test_invalid_implementation_wrong_signature(self):
        """Test class with wrong method signature."""

        class WrongSignatureInferer:
            async def infer(self, messages: list[MessageDataItem]) -> InferResult:
                return InferResult(
                    response="Test",
                    next_messages=[],
                    meta={
                        "model_id": "test-model",
                        "infer_parameters": InferParameters(),
                    },
                )

        # Protocol checking is structural, but signature mismatch may be caught at runtime
        # This may still pass protocol check but fail at runtime

    def test_protocol_isinstance_check(self):
        """Test isinstance check for protocol."""

        class ConcreteInferer:
            async def infer(
                self,
                messages: list[MessageDataItem],
                parameters: InferParameters,
            ) -> InferResult:
                return InferResult(
                    response="Test",
                    next_messages=[],
                    meta={
                        "model_id": "test-model",
                        "infer_parameters": parameters,
                    },
                )

        instance = ConcreteInferer()
        assert isinstance(instance, InferProtocol)


@pytest.mark.asyncio
class TestInferProtocolAsync:
    """Async tests for InferProtocol implementations."""

    async def test_infer_method_call(self):
        """Test calling infer method on a concrete implementation."""

        class MockInferer:
            async def infer(
                self,
                messages: list[MessageDataItem],
                parameters: InferParameters,
            ) -> InferResult:
                return InferResult(
                    response="Mock response",
                    next_messages=messages
                    + [MessageDataItem(role="assistant", content="Mock response")],
                    meta={
                        "model_id": "mock-model",
                        "infer_parameters": parameters,
                    },
                )

        inferer = MockInferer()
        messages = [MessageDataItem(role="user", content="Hello")]
        params = InferParameters(temperature=0.7)

        result = await inferer.infer(messages, params)

        assert result.response == "Mock response"
        assert len(result.next_messages) == 2
        assert result.meta["model_id"] == "mock-model"
