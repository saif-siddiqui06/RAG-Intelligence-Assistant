"""Generation: query rewriting and context-grounded answer generation.

- `base.py` — BaseChatModel, the provider-agnostic chat interface.
- `gemini_chat_model.py` — Gemini implementation of BaseChatModel.
- `prompts.py` — every system/user prompt template, kept in one file so
  what's actually sent to the LLM is easy to audit ("prompting" as its
  own concern, separate from retrieval and from the orchestration in
  app.services.chat_service).
- `query_rewriter.py` — LLM call #1: conversational question -> standalone,
  retrieval-optimized query.
- `answer_generator.py` — LLM call #2: rewritten query + retrieved chunks ->
  a grounded, cited answer (streaming or not), plus citation/confidence
  extraction helpers.

query_rewriter.py and answer_generator.py depend only on BaseChatModel,
never on Gemini's SDK directly — swapping providers again later means
adding one new class here, same as embeddings/vectorstore.
"""
