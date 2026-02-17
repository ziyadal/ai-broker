import glob
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import gradio as gr
from agents import Agent, Runner, function_tool
from agents.tracing import trace
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter

MODEL = "gpt-5.2-2025-12-11"
VECTOR_DB_DIR = "vector_db"
PROPERTIES_DB_PATH = "properties.db"
KB_ROOT = "knowledge-base"

retriever = None
agent = None
trace_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def bootstrap_environment() -> None:
    load_dotenv(override=True)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment or .env file.")


def validate_properties_db(path: str) -> None:
    db_path = Path(path)
    if not db_path.exists():
        raise FileNotFoundError(f"Properties database not found: {path}")

    required_columns = {
        "listing_id",
        "title",
        "price_aed",
        "bedrooms",
        "area",
        "handover",
        "image_url",
        "description",
    }

    conn = sqlite3.connect(path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prop_table'")
        if cursor.fetchone() is None:
            raise RuntimeError("Missing required table 'prop_table' in properties.db")

        cursor.execute("PRAGMA table_info(prop_table)")
        columns = {row[1] for row in cursor.fetchall()}
        missing = sorted(required_columns - columns)
        if missing:
            raise RuntimeError(f"prop_table is missing required columns: {', '.join(missing)}")
    finally:
        conn.close()


def load_kb_documents(root_dir: str) -> List[Any]:
    folders = glob.glob(os.path.join(root_dir, "*"))
    documents: List[Any] = []

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents


def split_documents_by_headers(documents: List[Any]) -> List[Any]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("###", "section")])
    chunks: List[Any] = []

    for doc in documents:
        split_docs = header_splitter.split_text(doc.page_content)
        for chunk in split_docs:
            chunk.metadata.update(doc.metadata)
        chunks.extend(split_docs)

    return chunks


def load_or_build_retriever() -> Any:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db_path = Path(VECTOR_DB_DIR)

    if db_path.exists():
        try:
            vectorstore = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
            if vectorstore._collection.count() > 0:
                return vectorstore.as_retriever()
        except Exception:
            pass

    documents = load_kb_documents(KB_ROOT)
    if not documents:
        raise RuntimeError(f"No markdown documents found under '{KB_ROOT}'.")

    chunks = split_documents_by_headers(documents)
    if not chunks:
        raise RuntimeError("KB splitting produced zero chunks. Check markdown content and headers.")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )
    return vectorstore.as_retriever()


@function_tool
def property_search(
    price: Optional[str] = None,
    rooms: Optional[int] = None,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search off-plan properties in the local SQLite DB.

    Optional filters:
    - price: max budget, numeric (supports 'm' shorthand such as '2.5m')
    - rooms: minimum bedroom count
    - location: area name (partial match)
    """
    max_price: Optional[int] = None
    if price is not None and str(price).strip():
        p = str(price).lower().replace(",", "").strip()
        if "m" in p:
            max_price = int(float(p.replace("m", "").strip()) * 1_000_000)
        else:
            max_price = int(p)

    where_clauses: List[str] = []
    params: List[Any] = []

    if max_price is not None:
        where_clauses.append("price_aed <= ?")
        params.append(max_price)

    if rooms is not None:
        where_clauses.append("bedrooms >= ?")
        params.append(int(rooms))

    if location is not None and str(location).strip():
        where_clauses.append("LOWER(area) LIKE LOWER(?)")
        params.append(f"%{str(location).strip()}%")

    if not where_clauses:
        return {"ok": False, "error": "Provide at least one filter: price, rooms, or location."}

    where_sql = " AND ".join(where_clauses)

    if max_price is not None:
        order_sql = "ORDER BY ABS(price_aed - ?) ASC"
        params_for_query = params + [max_price]
    else:
        order_sql = "ORDER BY price_aed ASC"
        params_for_query = params

    query = f"""
    SELECT listing_id, title, price_aed, bedrooms, area, handover, image_url, description
    FROM prop_table
    WHERE {where_sql}
    {order_sql}
    LIMIT 7
    """

    conn = sqlite3.connect(PROPERTIES_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(query, params_for_query)
        rows = cursor.fetchall()
    finally:
        conn.close()

    results = []
    for listing_id, title, price_aed, bedrooms, area, handover, image_url, description in rows:
        results.append(
            {
                "listing_id": listing_id,
                "title": title,
                "price_aed": price_aed,
                "bedrooms": bedrooms,
                "area": area,
                "handover": handover,
                "image_url": image_url,
                "description": description,
            }
        )

    if not results:
        return {"ok": True, "message": "No properties found matching the criteria.", "results": []}

    return {"ok": True, "results": results}


@function_tool
def search_docs(question: str) -> str:
    """
    Retrieve KB content for UAE real-estate legal, visa, tax, financing, and ownership questions.
    Query should be in English.
    """
    global retriever
    if retriever is None:
        return "Knowledge base is not initialized yet. Please try again in a moment."

    try:
        docs = retriever.invoke(question)
    except Exception as exc:
        return f"Knowledge retrieval error: {exc}"

    if not docs:
        return "No relevant KB passages found. Ask a more specific question."

    return "\n\n".join(getattr(doc, "page_content", "") for doc in docs if getattr(doc, "page_content", ""))


def build_agent() -> Agent:
    instructions = """
You are a knowledgeable and concise UAE off-plan real-estate broker assistant for foreign investors.

Tool usage policy:
- Use `property_search` when user asks for recommendations, budget/rooms/location matching, or specific properties.
- Use `search_docs` for legal/regulatory/tax/visa/financing/ownership facts.
- For factual/legal answers, rely on `search_docs` and do not guess.
- If no docs are found, ask a clarifying follow-up question.
- If property search has no matches, ask the user to adjust budget/rooms/location naturally (do not mention tool failure).
- Recommend at most 3 properties in any single response.
""".strip()

    return Agent(
        name="Broker Agent",
        instructions=instructions,
        tools=[property_search, search_docs],
        model=MODEL,
    )


def _extract_properties_from_result(run_result: Any) -> List[Dict[str, Any]]:
    properties: List[Dict[str, Any]] = []

    for item in getattr(run_result, "new_items", []) or []:
        output = getattr(item, "output", None)
        if isinstance(output, dict) and isinstance(output.get("results"), list):
            properties = output["results"][:3]

    return properties[:3]


def _safe_price(value: Any) -> str:
    try:
        return f"AED {int(value):,}"
    except (TypeError, ValueError):
        return "AED N/A"


def _format_property_text(prop: Dict[str, Any]) -> str:
    description = str(prop.get("description") or "No description available.").strip()
    return (
        f"ID: {prop.get('listing_id', 'N/A')}\n"
        f"Title: {prop.get('title', 'Untitled property')}\n"
        f"Price: {_safe_price(prop.get('price_aed'))}\n"
        f"Bedrooms: {prop.get('bedrooms', 'N/A')}\n"
        f"Area: {prop.get('area', 'N/A')}\n"
        f"Handover: {prop.get('handover', 'N/A')}\n\n"
        f"Description:\n{description}"
    )


def _build_property_outputs(properties: List[Dict[str, Any]]) -> tuple[List[Optional[str]], List[str]]:
    images: List[Optional[str]] = [None, None, None]
    texts: List[str] = ["", "", ""]

    for idx, prop in enumerate(properties[:3]):
        image_url = prop.get("image_url")
        images[idx] = str(image_url) if image_url else None
        texts[idx] = _format_property_text(prop)

    return images, texts


def normalize_part_type(role: str, part: Dict[str, Any]) -> Dict[str, str]:
    text_value = str(part.get("text", ""))
    if role == "user":
        return {"type": "input_text", "text": text_value}
    return {"type": "output_text", "text": text_value}


def to_responses_input(history: Any, new_user_message: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                items.append({"role": "user", "content": [{"type": "input_text", "text": str(user_msg)}]})
            if assistant_msg:
                items.append({"role": "assistant", "content": [{"type": "output_text", "text": str(assistant_msg)}]})
            continue

        if isinstance(item, dict) and "role" in item:
            role = item["role"]
            content = item.get("content", "")

            if isinstance(content, list):
                fixed_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("text", "input_text", "output_text"):
                        fixed_parts.append(normalize_part_type(role, part))
                if fixed_parts:
                    items.append({"role": role, "content": fixed_parts})
            elif isinstance(content, str):
                part_type = "input_text" if role == "user" else "output_text"
                items.append({"role": role, "content": [{"type": part_type, "text": content}]})

    items.append({"role": "user", "content": [{"type": "input_text", "text": new_user_message}]})
    return items


async def agent_chat(message: str, history: Any):
    global agent
    if agent is None:
        return (
            "Agent is not initialized.",
            None,
            "",
            None,
            "",
            None,
            "",
        )

    inp = to_responses_input(history, message)

    with trace(trace_id):
        result = await Runner.run(agent, input=inp)

    properties = _extract_properties_from_result(result)
    images, texts = _build_property_outputs(properties)

    return (
        str(result.final_output),
        images[0],
        texts[0],
        images[1],
        texts[1],
        images[2],
        texts[2],
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks() as demo:
        prop1_img = gr.Image(label="Property 1", render=False)
        prop1_desc = gr.Textbox(label="Description 1", lines=10, interactive=False, render=False)

        prop2_img = gr.Image(label="Property 2", render=False)
        prop2_desc = gr.Textbox(label="Description 2", lines=10, interactive=False, render=False)

        prop3_img = gr.Image(label="Property 3", render=False)
        prop3_desc = gr.Textbox(label="Description 3", lines=10, interactive=False, render=False)

        gr.ChatInterface(
            fn=agent_chat,
            title="Broker Agent",
            additional_outputs=[prop1_img, prop1_desc, prop2_img, prop2_desc, prop3_img, prop3_desc],
        )

        with gr.Accordion("Recommended properties", open=True):
            with gr.Row():
                with gr.Column():
                    prop1_img.render()
                    prop1_desc.render()
                with gr.Column():
                    prop2_img.render()
                    prop2_desc.render()
                with gr.Column():
                    prop3_img.render()
                    prop3_desc.render()

    return demo


def main() -> None:
    global retriever, agent

    bootstrap_environment()
    validate_properties_db(PROPERTIES_DB_PATH)
    retriever = load_or_build_retriever()
    agent = build_agent()

    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
