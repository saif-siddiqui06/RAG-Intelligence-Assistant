"""Tests for LLM-based query rewriting using bounded conversation history."""
from app.rag.generation.query_rewriter import QueryRewriter
from tests.fakes import FakeChatModel


def test_no_history_returns_question_unchanged_without_calling_llm():
    chat_model = FakeChatModel(responses=["should never be used"])
    rewriter = QueryRewriter(chat_model)

    result = rewriter.rewrite(history=[], question="What is SMOTE?")

    assert result == "What is SMOTE?"
    assert chat_model.calls == []


def test_with_history_calls_llm_and_returns_rewritten_query():
    chat_model = FakeChatModel(responses=["What are the disadvantages of SMOTE?"])
    rewriter = QueryRewriter(chat_model)

    history = [("user", "What is SMOTE?"), ("assistant", "SMOTE is an oversampling technique.")]
    result = rewriter.rewrite(history, "What are its disadvantages?")

    assert result == "What are the disadvantages of SMOTE?"
    assert len(chat_model.calls) == 1


def test_rewrite_prompt_includes_conversation_history():
    chat_model = FakeChatModel(responses=["What are the disadvantages of SMOTE?"])
    rewriter = QueryRewriter(chat_model)

    history = [("user", "What is SMOTE?"), ("assistant", "SMOTE is an oversampling technique.")]
    rewriter.rewrite(history, "What are its disadvantages?")

    user_message = chat_model.calls[0][-1]["content"]
    assert "What is SMOTE?" in user_message
    assert "its disadvantages" in user_message


def test_llm_failure_falls_back_to_the_original_question():
    class BoomChatModel:
        def complete(self, messages, temperature=0):
            raise RuntimeError("network down")

        def stream(self, messages, temperature=0):
            raise RuntimeError("network down")

    rewriter = QueryRewriter(BoomChatModel())
    result = rewriter.rewrite(history=[("user", "hi")], question="What are its disadvantages?")

    assert result == "What are its disadvantages?"


def test_blank_llm_response_falls_back_to_the_original_question():
    chat_model = FakeChatModel(responses=["   "])
    rewriter = QueryRewriter(chat_model)

    result = rewriter.rewrite(history=[("user", "hi")], question="original question")

    assert result == "original question"
