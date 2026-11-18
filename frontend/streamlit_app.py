import os
from typing import Any

import requests
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = 60


class BackendAPIError(Exception):
    """Raised when the FastAPI backend returns an error or is unreachable."""


def format_source_label(source: dict[str, Any]) -> str:
    filename = source.get("filename", "Unknown Source")
    page_number = source.get("page_number")
    source_url = source.get("source_url")
    score = source.get("score")

    loc = f"Page {page_number}" if page_number is not None else (source_url or "Web")
    if score is not None:
        return f"{filename} — {loc} (score: {score:.2f})"
    return f"{filename} — {loc}"


def fetch_documents(api_base_url: str = DEFAULT_API_BASE_URL) -> list[dict[str, Any]]:
    try:
        res = requests.get(f"{api_base_url}/documents", timeout=15)
        res.raise_for_status()
        data = res.json()
        return list(data.get("documents", []))
    except requests.RequestException as exc:
        raise BackendAPIError(f"Could not connect to backend at {api_base_url}: {exc}") from exc


def upload_pdf_to_api(
    file_name: str,
    file_bytes: bytes,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    try:
        res = requests.post(
            f"{api_base_url}/documents/upload",
            files={"file": (file_name, file_bytes, "application/pdf")},
            timeout=REQUEST_TIMEOUT,
        )
        if res.status_code >= 400:
            detail = res.json().get("detail", res.text)
            raise BackendAPIError(f"Upload failed: {detail}")
        return dict(res.json())
    except requests.RequestException as exc:
        raise BackendAPIError(f"Network error during PDF upload: {exc}") from exc


def ingest_url_to_api(
    url: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    try:
        res = requests.post(
            f"{api_base_url}/documents/url",
            json={"url": url},
            timeout=REQUEST_TIMEOUT,
        )
        if res.status_code >= 400:
            detail = res.json().get("detail", res.text)
            raise BackendAPIError(f"URL ingestion failed: {detail}")
        return dict(res.json())
    except requests.RequestException as exc:
        raise BackendAPIError(f"Network error during URL ingestion: {exc}") from exc


def delete_document_from_api(
    document_id: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> bool:
    try:
        res = requests.delete(f"{api_base_url}/documents/{document_id}", timeout=15)
        if res.status_code >= 400:
            detail = res.json().get("detail", res.text)
            raise BackendAPIError(f"Delete failed: {detail}")
        return bool(res.json().get("deleted", False))
    except requests.RequestException as exc:
        raise BackendAPIError(f"Network error while deleting document: {exc}") from exc


def ask_question_to_api(
    question: str,
    *,
    document_ids: list[str] | None = None,
    history: list[dict[str, str]] | None = None,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": question,
        "history": history or [],
    }
    if document_ids:
        payload["document_ids"] = document_ids

    try:
        res = requests.post(
            f"{api_base_url}/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        if res.status_code >= 400:
            detail = res.json().get("detail", res.text)
            raise BackendAPIError(f"Chat request failed: {detail}")
        return dict(res.json())
    except requests.RequestException as exc:
        raise BackendAPIError(f"Network error while asking question: {exc}") from exc


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for idx, src in enumerate(sources, start=1):
            label = format_source_label(src)
            st.markdown(f"**{idx}. {label}**")
            source_url = src.get("source_url")
            if source_url:
                st.markdown(f"🔗 [{source_url}]({source_url})")
            snippet = src.get("snippet", "")
            if snippet:
                st.caption(snippet)
            if idx < len(sources):
                st.divider()


def main() -> None:
    st.set_page_config(
        page_title="Document Intelligence RAG",
        page_icon="📄",
        layout="wide",
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    st.title("📄 Document Intelligence RAG")
    st.caption(
        "Upload PDFs or ingest websites, then ask grounded natural-language questions "
        "with verified citations powered by Google Gemini and pgvector."
    )

    selected_doc_ids: list[str] = []

    with st.sidebar:
        st.header("📥 Add Knowledge Sources")

        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        if st.button("Upload & Index PDF", use_container_width=True, disabled=not uploaded_file):
            if uploaded_file is not None:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    try:
                        doc_res = upload_pdf_to_api(
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                        )
                        if doc_res.get("deduplicated"):
                            st.info(
                                f"'{doc_res['filename']}' was already indexed "
                                f"({doc_res['chunk_count']} chunks)."
                            )
                        else:
                            st.success(
                                f"Indexed '{doc_res['filename']}' "
                                f"({doc_res['chunk_count']} chunks)!"
                            )
                    except BackendAPIError as exc:
                        st.error(str(exc))

        st.divider()
        st.subheader("Add Website")
        url_input = st.text_input(
            "Public Webpage URL",
            placeholder="https://example.com/article",
        )
        if st.button("Ingest Webpage", use_container_width=True, disabled=not url_input.strip()):
            with st.spinner("Fetching and indexing webpage..."):
                try:
                    doc_res = ingest_url_to_api(url_input.strip())
                    if doc_res.get("deduplicated"):
                        st.info(
                            f"'{doc_res['filename']}' was already indexed "
                            f"({doc_res['chunk_count']} chunks)."
                        )
                    else:
                        st.success(
                            f"Indexed '{doc_res['filename']}' "
                            f"({doc_res['chunk_count']} chunks)!"
                        )
                except BackendAPIError as exc:
                    st.error(str(exc))

        st.divider()
        st.header("🗂️ Indexed Documents")

        docs: list[dict[str, Any]] = []
        try:
            docs = fetch_documents()
        except BackendAPIError as exc:
            st.warning(str(exc))

        completed_docs = [d for d in docs if d.get("status") == "completed"]
        doc_options = {
            f"{d['filename']} ({d.get('chunk_count', 0)} chunks)": str(d["id"])
            for d in completed_docs
        }

        if doc_options:
            selected_labels = st.multiselect(
                "Scope chat to specific documents (optional):",
                options=list(doc_options.keys()),
                help="Leave empty to search across all indexed documents.",
            )
            selected_doc_ids = [doc_options[label] for label in selected_labels]

        if not docs:
            st.caption("No documents indexed yet. Upload a PDF or add a URL above.")
        else:
            for doc in docs:
                status_icon = {
                    "completed": "✅",
                    "processing": "⏳",
                    "failed": "❌",
                }.get(doc.get("status", ""), "📄")
                col_info, col_del = st.columns([4, 1])
                with col_info:
                    st.markdown(
                        f"{status_icon} **{doc['filename']}**  \n"
                        f"<small>{doc['source_type'].upper()} • "
                        f"{doc.get('chunk_count', 0)} chunks</small>",
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_{doc['id']}", help=f"Delete {doc['filename']}"):
                        try:
                            delete_document_from_api(str(doc["id"]))
                            st.rerun()
                        except BackendAPIError as exc:
                            st.error(str(exc))

        st.divider()
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    # Render conversation history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    # Chat input
    user_question = st.chat_input("Ask a question about your indexed documents...")
    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        prior_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"]
        ]
        st.session_state["messages"].append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating grounded answer..."):
                try:
                    response_data = ask_question_to_api(
                        user_question,
                        document_ids=selected_doc_ids or None,
                        history=prior_history,
                    )
                    answer_text = response_data.get("answer", "")
                    sources_list = response_data.get("sources", [])
                    st.markdown(answer_text)
                    render_sources(sources_list)

                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": answer_text,
                            "sources": sources_list,
                        }
                    )
                except BackendAPIError as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    main()
