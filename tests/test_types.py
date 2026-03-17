"""Tests for judger.types module."""

import pytest
from pydantic import ValidationError

from judger.types import (
    InferMeta,
    InferParameters,
    InferResult,
    InputItem,
    JudgeResult,
    MessageDataItem,
    _JudgeMeta,
)


class TestMessageDataItem:
    """Tests for MessageDataItem model."""

    def test_create_with_valid_role_user(self):
        """Test creating MessageDataItem with user role."""
        item = MessageDataItem(role="user", content="Hello")
        assert item.role == "user"
        assert item.content == "Hello"

    def test_create_with_valid_role_assistant(self):
        """Test creating MessageDataItem with assistant role."""
        item = MessageDataItem(role="assistant", content="Hi there")
        assert item.role == "assistant"
        assert item.content == "Hi there"

    def test_create_with_valid_role_system(self):
        """Test creating MessageDataItem with system role."""
        item = MessageDataItem(role="system", content="You are helpful")
        assert item.role == "system"
        assert item.content == "You are helpful"

    def test_invalid_role_rejected(self):
        """Test that invalid role values are rejected."""
        with pytest.raises(ValidationError):
            MessageDataItem(role="invalid", content="Hello")

    def test_empty_content_allowed(self):
        """Test that empty content is allowed."""
        item = MessageDataItem(role="user", content="")
        assert item.content == ""


class TestInferParameters:
    """Tests for InferParameters model."""

    def test_default_values(self):
        """Test default parameter values."""
        params = InferParameters()
        assert params.top_p == 0.95
        assert params.temperature == 1.0

    def test_custom_values(self):
        """Test custom parameter values."""
        params = InferParameters(top_p=0.9, temperature=0.7)
        assert params.top_p == 0.9
        assert params.temperature == 0.7

    def test_partial_custom_values(self):
        """Test setting only some custom values."""
        params = InferParameters(temperature=0.5)
        assert params.top_p == 0.95  # default
        assert params.temperature == 0.5


class TestInferResult:
    """Tests for InferResult model."""

    def test_create_infer_result(self):
        """Test creating InferResult with all fields."""
        message = MessageDataItem(role="assistant", content="Response")
        meta: InferMeta = {
            "model_id": "gpt-4",
            "infer_parameters": InferParameters(),
        }
        result = InferResult(
            response="Hello world",
            next_messages=[message],
            meta=meta,
        )
        assert result.response == "Hello world"
        assert len(result.next_messages) == 1
        assert result.meta["model_id"] == "gpt-4"

    def test_infer_result_with_empty_messages(self):
        """Test creating InferResult with empty next_messages."""
        meta: InferMeta = {
            "model_id": "model-1",
            "infer_parameters": InferParameters(top_p=0.9),
        }
        result = InferResult(
            response="Test",
            next_messages=[],
            meta=meta,
        )
        assert result.next_messages == []


class TestInputItem:
    """Tests for InputItem model."""

    def test_create_with_required_fields(self):
        """Test creating InputItem with required fields only."""
        item = InputItem(
            id="test-1",
            question="What is AI?",
            response="AI is artificial intelligence.",
            meta={},
        )
        assert item.id == "test-1"
        assert item.question == "What is AI?"
        assert item.response == "AI is artificial intelligence."
        assert item.ref_answer is None
        assert item.meta == {}

    def test_create_with_optional_ref_answer(self):
        """Test creating InputItem with ref_answer."""
        item = InputItem(
            id="test-2",
            question="What is Python?",
            response="A programming language.",
            ref_answer="Python is a high-level programming language.",
            meta={"category": "programming"},
        )
        assert item.ref_answer == "Python is a high-level programming language."
        assert item.meta["category"] == "programming"

    def test_create_with_complex_meta(self):
        """Test creating InputItem with complex meta data."""
        item = InputItem(
            id="test-3",
            question="Calculate 2+2",
            response="4",
            meta={
                "difficulty": "easy",
                "tags": ["math", "basic"],
                "score": 10,
            },
        )
        assert item.meta["difficulty"] == "easy"
        assert item.meta["tags"] == ["math", "basic"]
        assert item.meta["score"] == 10


class TestJudgeResult:
    """Tests for JudgeResult model."""

    def test_create_judge_result(self):
        """Test creating JudgeResult with all fields."""
        meta: _JudgeMeta = {
            "question": "What is AI?",
            "response": "AI is technology.",
            "ref_answer": "AI stands for Artificial Intelligence.",
        }
        result = JudgeResult(
            id="judge-1",
            score=0.85,
            reason="Good answer but missing full expansion.",
            meta=meta,
        )
        assert result.id == "judge-1"
        assert result.score == 0.85
        assert result.reason == "Good answer but missing full expansion."
        assert result.meta["question"] == "What is AI?"

    def test_judge_result_with_zero_score(self):
        """Test creating JudgeResult with zero score."""
        meta: _JudgeMeta = {
            "question": "Question?",
            "response": "Wrong answer.",
            "ref_answer": "Correct answer.",
        }
        result = JudgeResult(
            id="judge-2",
            score=0.0,
            reason="Incorrect answer.",
            meta=meta,
        )
        assert result.score == 0.0

    def test_judge_result_with_perfect_score(self):
        """Test creating JudgeResult with perfect score."""
        meta: _JudgeMeta = {
            "question": "Question?",
            "response": "Perfect answer.",
            "ref_answer": "Perfect answer.",
        }
        result = JudgeResult(
            id="judge-3",
            score=1.0,
            reason="Perfect match.",
            meta=meta,
        )
        assert result.score == 1.0


class TestTypeValidation:
    """Tests for type validation across models."""

    def test_score_must_be_float(self):
        """Test that score field accepts numeric values."""
        meta: _JudgeMeta = {
            "question": "Q",
            "response": "R",
            "ref_answer": "A",
        }
        # Pydantic v2 coerces int to float
        result = JudgeResult(
            id="test",
            score=1,  # int
            reason="Test",
            meta=meta,
        )
        assert isinstance(result.score, float)
        assert result.score == 1.0

    def test_id_must_be_string(self):
        """Test that id field requires string."""
        with pytest.raises(ValidationError):
            InputItem(
                id=123,  # Should be string
                question="Q",
                response="R",
                meta={},
            )
