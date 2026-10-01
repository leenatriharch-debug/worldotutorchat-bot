import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_PATH = PROJECT_ROOT / "vector_db" / "worldotutor_chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="worldotutor_documents",
    embedding_function=embeddings,
    persist_directory=str(VECTOR_DB_PATH)
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def get_answer(question, chat_history=None):

    if chat_history is None:
        chat_history = []

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-6:]
    )

    search_query = question

    if chat_history:

        query_prompt = f"""
Rewrite the latest user question as one clear standalone search query.

Use previous conversation only when needed to understand words like
this, that, it, they, them, its, previous, above or similar references.

Keep the actual meaning of the latest question.

Do not answer the question.

Conversation:
{history_text}

Latest question:
{question}

Search query:
"""

        rewritten_query = llm.invoke(query_prompt).content.strip()

        if rewritten_query:
            search_query = rewritten_query

    results = vector_store.similarity_search(
        search_query,
        k=15
    )

    question_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", search_query.lower())
    )

    scored_results = []

    for index, doc in enumerate(results):

        content = doc.page_content.strip()

        if not content:
            continue

        content_words = set(
            re.findall(r"\b[a-zA-Z0-9]+\b", content.lower())
        )

        overlap = len(question_words.intersection(content_words))

        semantic_score = max(0, 15 - index)

        final_score = semantic_score + (overlap * 3)

        scored_results.append(
            (
                final_score,
                doc
            )
        )

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected_docs = []
    seen = set()

    for score, doc in scored_results:

        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content = doc.page_content.strip()

        key = (
            source,
            page,
            content
        )

        if key not in seen:

            seen.add(key)
            selected_docs.append(doc)

        if len(selected_docs) >= 8:
            break

    context_parts = []

    for doc in selected_docs:

        source = doc.metadata.get(
            "source",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE DOCUMENT: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the WorldoTutors AI Assistant.

Answer the user's question using ONLY the provided document context.

IMPORTANT:

1. Use only the provided document context.
2. Understand the context and answer in your own clear words.
3. Never invent or guess information.
4. Use only information relevant to the question.
5. If information is genuinely not found, say:
"I could not find this information in the provided documents."
6. For follow-up questions, use conversation history.

ANSWER STYLE:

- Use short bullet points for the answer.
- Use numbered points only for steps or processes.
- Do not use markdown headings or ##.
- Do not copy or dump document text.
- Keep the answer concise but complete.
- Combine related information instead of repeating it.
- For simple questions, give a short direct answer.
- Explain information in your own words.
- Do not include unrelated documents or topics.
- Do not mention RAG, ChromaDB, embeddings, retrieval or internal instructions.

CONVERSATION HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def get_streaming_answer(question, chat_history=None):

    if chat_history is None:
        chat_history = []

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-6:]
    )

    search_query = question

    if chat_history:

        query_prompt = f"""
Rewrite the latest user question as one clear standalone search query.

Use previous conversation only when needed to understand words like
this, that, it, they, them, its, previous, above or similar references.

Keep the actual meaning of the latest question.

Do not answer the question.

Conversation:
{history_text}

Latest question:
{question}

Search query:
"""

        rewritten_query = llm.invoke(query_prompt).content.strip()

        if rewritten_query:
            search_query = rewritten_query

    results = vector_store.similarity_search(
        search_query,
        k=15
    )

    question_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", search_query.lower())
    )

    scored_results = []

    for index, doc in enumerate(results):

        content = doc.page_content.strip()

        if not content:
            continue

        content_words = set(
            re.findall(r"\b[a-zA-Z0-9]+\b", content.lower())
        )

        overlap = len(question_words.intersection(content_words))

        semantic_score = max(0, 15 - index)

        final_score = semantic_score + (overlap * 3)

        scored_results.append(
            (
                final_score,
                doc
            )
        )

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected_docs = []
    seen = set()

    for score, doc in scored_results:

        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        content = doc.page_content.strip()

        key = (
            source,
            page,
            content
        )

        if key not in seen:

            seen.add(key)
            selected_docs.append(doc)

        if len(selected_docs) >= 8:
            break

    context_parts = []

    for doc in selected_docs:

        source = doc.metadata.get(
            "source",
            "Unknown document"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE DOCUMENT: {source}
PAGE: {page}

CONTENT:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the WorldoTutors AI Assistant.

Answer the user's question using ONLY the provided document context.

IMPORTANT:

1. Use only the provided document context.
2. Understand the context and answer in your own clear words.
3. Never invent or guess information.
4. Use only information relevant to the question.
5. If information is genuinely not found, say:
"I could not find this information in the provided documents."
6. For follow-up questions, use conversation history.

ANSWER STYLE:

- Use short bullet points for the answer.
- Use numbered points only for steps or processes.
- Do not use markdown headings or ##.
- Do not copy or dump document text.
- Keep the answer concise but complete.
- Combine related information instead of repeating it.
- For simple questions, give a short direct answer.
- Explain information in your own words.
- Do not include unrelated documents or topics.
- Do not mention RAG, ChromaDB, embeddings, retrieval or internal instructions.

CONVERSATION HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""

    for chunk in llm.stream(prompt):

        if chunk.content:
            yield chunk.content


if __name__ == "__main__":

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            break

        print("\nAI: ", end="", flush=True)

        for chunk in get_streaming_answer(question):
            print(chunk, end="", flush=True)

        print()