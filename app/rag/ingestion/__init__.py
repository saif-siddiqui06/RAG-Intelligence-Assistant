"""Document ingestion: PDF extraction, cleaning, chunking and file
hashing. Deliberately independent from any future retrieval/chat
pipeline — this package only turns files into (metadata, text) chunks
ready to embed; it knows nothing about answering questions.
"""
