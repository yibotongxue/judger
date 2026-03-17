"""Tests for judger.judger.judger_protocol module."""

from typing import get_type_hints

import pytest

from judger.judger.judger_protocol import JudgeProtocol
from judger.types import InputItem, JudgeResult


class TestJudgeProtocol:
    """Tests for JudgeProtocol class."""

    def test_judge_protocol_is_runtime_checkable(self):
        """Test that JudgeProtocol is decorated with @runtime_checkable."""
        # Check if the protocol is runtime checkable
        assert hasattr(JudgeProtocol, "_is_runtime_protocol")
        assert JudgeProtocol._is_runtime_protocol is True

    def test_judge_protocol_is_abstract(self):
        """Test that JudgeProtocol cannot be instantiated directly."""
        with pytest.raises(TypeError):
            JudgeProtocol()

    def test_judge_method_signature(self):
        """Test that judge method has correct signature."""
        # Check method exists
        assert hasattr(JudgeProtocol, "judge")

        # Check it's a method (function attribute)
        assert callable(getattr(JudgeProtocol, "judge"))

    def test_judge_method_parameters(self):
        """Test judge method parameter types."""
        hints = get_type_hints(JudgeProtocol.judge)

        # Check parameter types
        assert hints["inputs"] == list[InputItem]
        assert hints["return"] == list[JudgeResult]


class TestJudgeProtocolImplementation:
    """Tests for JudgeProtocol implementations."""

    def test_valid_implementation(self):
        """Test a valid implementation of JudgeProtocol."""

        class ValidJudger:
            async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return [
                    JudgeResult(
                        id=item.id,
                        score=1.0,
                        reason="Good",
                        meta={
                            "question": item.question,
                            "response": item.response,
                            "ref_answer": item.ref_answer or "",
                        },
                    )
                    for item in inputs
                ]

        # Check that ValidJudger implements the protocol
        assert issubclass(ValidJudger, JudgeProtocol)

    def test_invalid_implementation_missing_method(self):
        """Test class missing judge method doesn't implement protocol."""

        class InvalidJudger:
            pass

        # This class doesn't implement JudgeProtocol
        assert not issubclass(InvalidJudger, JudgeProtocol)

    def test_invalid_implementation_wrong_return_type(self):
        """Test class with wrong return type."""

        class WrongReturnJudger:
            async def judge(self, inputs: list[InputItem]) -> JudgeResult:
                # Returns single result instead of list
                return JudgeResult(
                    id="test",
                    score=1.0,
                    reason="Test",
                    meta={
                        "question": "Q",
                        "response": "R",
                        "ref_answer": "A",
                    },
                )

        # Protocol checking is structural, may pass at check time but fail at runtime
        # For runtime_checkable, the method signature must match

    def test_protocol_isinstance_check(self):
        """Test isinstance check for protocol."""

        class ConcreteJudger:
            async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return [
                    JudgeResult(
                        id=item.id,
                        score=0.5,
                        reason="Partial",
                        meta={
                            "question": item.question,
                            "response": item.response,
                            "ref_answer": item.ref_answer or "",
                        },
                    )
                    for item in inputs
                ]

        instance = ConcreteJudger()
        assert isinstance(instance, JudgeProtocol)

    def test_non_implementation_isinstance_check(self):
        """Test isinstance returns False for non-implementing class."""

        class NotAJudger:
            pass

        instance = NotAJudger()
        assert not isinstance(instance, JudgeProtocol)


@pytest.mark.asyncio
class TestJudgeProtocolAsync:
    """Async tests for JudgeProtocol implementations."""

    async def test_judge_method_call_empty_list(self):
        """Test calling judge method with empty list."""

        class MockJudger:
            async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return []

        judger = MockJudger()
        result = await judger.judge([])

        assert result == []

    async def test_judge_method_call_with_items(self):
        """Test calling judge method with actual items."""

        class MockJudger:
            async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return [
                    JudgeResult(
                        id=item.id,
                        score=1.0,
                        reason="Mock reason",
                        meta={
                            "question": item.question,
                            "response": item.response,
                            "ref_answer": item.ref_answer or "",
                        },
                    )
                    for item in inputs
                ]

        judger = MockJudger()
        inputs = [
            InputItem(
                id="test-1",
                question="Q1?",
                response="A1",
                ref_answer="Ref1",
                meta={},
            ),
            InputItem(
                id="test-2",
                question="Q2?",
                response="A2",
                meta={},
            ),
        ]

        results = await judger.judge(inputs)

        assert len(results) == 2
        assert results[0].id == "test-1"
        assert results[0].score == 1.0
        assert results[0].reason == "Mock reason"
        assert results[0].meta["question"] == "Q1?"
        assert results[1].id == "test-2"
        assert results[1].meta["ref_answer"] == ""


class TestProtocolEdgeCases:
    """Edge case tests for JudgeProtocol."""

    def test_protocol_with_extra_methods(self):
        """Test that class with extra methods still implements protocol."""

        class ExtendedJudger:
            async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return []

            async def batch_judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return []

            def configure(self, config: dict) -> None:
                pass

        assert issubclass(ExtendedJudger, JudgeProtocol)

    def test_protocol_inheritance(self):
        """Test protocol implementation through inheritance."""

        class BaseJudger:
            async def judge(self, inputs: list[InputItem]) -> list[JudgeResult]:
                return []

        class DerivedJudger(BaseJudger):
            pass

        assert issubclass(DerivedJudger, JudgeProtocol)
        assert isinstance(DerivedJudger(), JudgeProtocol)
