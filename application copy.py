import glob
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from agents import Agent, Runner, function_tool
from agents.tracing import trace
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from nicegui import ui

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
    Retrieve Knowledge base content for UAE real-estate legal, visa, tax, financing, and ownership questions.
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
You are a friendly and knowledgable senior off-plan real-estate broker who specialises in helping foreign investors invest in the UAE.

You help users identify high-quality real estate opportunities based on their budget, goals, and preferences.

You provide clear, concise, and investor-focused recommendations.

Your goal is to:
- Understand the user's requirements
- Retrieve relevant property options
- Recommend the best matches
- Clearly explain WHY each option is suitable
- Keep responses concise and structured


When recommending properties, ALWAYS follow this structure:

[Property Name / Area]
- Price:
- Bedrooms:
- Handover:
- Key Investment Reason:
- Why it fits YOU:

After listing properties, include:

Summary:
(1–2 lines explaining overall recommendation or strategy)

Tool usage policy:
- Use `property_search` tool to search for avaiable properties that can be recommended to the user. 
- Use `search_docs` for legal/regulatory/tax/visa/financing/ownership facts. Only include information that directly relates to the user's question.
- For factual/legal answers, rely on `search_docs` and do not guess.
- If more information is required ask a clarifying follow-up questions.
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
    response_text, properties = await run_agent_turn(message, history)
    images, texts = _build_property_outputs(properties)

    return (
        response_text,
        images[0],
        texts[0],
        images[1],
        texts[1],
        images[2],
        texts[2],
    )


async def run_agent_turn(message: str, history: Any) -> tuple[str, List[Dict[str, Any]]]:
    global agent
    if agent is None:
        return "Agent is not initialized.", []

    inp = to_responses_input(history, message)
    with trace(trace_id):
        result = await Runner.run(agent, input=inp)

    return str(result.final_output), _extract_properties_from_result(result)


def _clip_text(value: Any, max_len: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_len:
        return text
    return f"{text[:max_len].rstrip()}..."


def _inject_styles() -> None:
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
          :root {
            --primary: #1b1a18;
            --secondary: #4f4a43;
            --accent: #b7935a;
            --bg-ink: #f7f4ef;
            --bg-night: #eee8df;
            --surface: rgba(255, 255, 255, 0.97);
            --surface-strong: #ffffff;
            --text-main: #1f1e1c;
            --text-soft: #58534c;
            --line: rgba(27, 26, 24, 0.14);
            --accent-soft: rgba(183, 147, 90, 0.2);
            --gold: #b7935a;
          }

          html,
          body {
            margin: 0;
            height: 100%;
            overflow-y: auto;
            overflow-x: hidden;
          }

          body {
            font-family: "Manrope", sans-serif;
            color: var(--text-main);
            background:
              radial-gradient(circle at 10% 8%, rgba(183, 147, 90, 0.2), transparent 34%),
              radial-gradient(circle at 92% 5%, rgba(27, 26, 24, 0.06), transparent 24%),
              linear-gradient(150deg, var(--bg-ink), var(--bg-night));
          }

          .nicegui-content {
            padding: 0 !important;
            min-height: 100vh;
            overflow-x: hidden;
            overflow-y: auto;
          }

          .app-shell {
            width: min(1420px, 96vw);
            margin: 0 auto;
            padding: 0.62rem 0;
            gap: 0.62rem;
            min-height: 100vh;
            box-sizing: border-box;
            overflow: visible;
          }

          .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 0.82rem 1.08rem;
            border-radius: 16px;
            border: 1px solid rgba(183, 147, 90, 0.4);
            background: linear-gradient(125deg, rgba(32, 30, 27, 0.95), rgba(52, 45, 38, 0.95));
            box-shadow: 0 14px 34px rgba(17, 17, 17, 0.14);
            animation: rise .55s ease;
            flex: 0 0 auto;
          }

          .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background: linear-gradient(110deg, transparent 0%, rgba(183, 147, 90, 0.18) 48%, transparent 100%);
            transform: translateX(-120%);
            animation: sweep 2.4s ease-out 0.35s 1;
          }

          .hero-title {
            font-family: "Cormorant Garamond", serif;
            font-size: clamp(1.82rem, 2.6vw, 2.6rem);
            font-weight: 600;
            line-height: 1.05;
            letter-spacing: 0.4px;
            color: #f6f0e6;
          }

          .hero-subtitle {
            margin-top: 0.16rem;
            color: #e3d5bf;
            font-size: 0.92rem;
          }

          .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.7rem;
            width: 100%;
            flex: 1 1 auto;
            min-height: 0;
          }

          .panel {
            border-radius: 20px;
            border: 1px solid var(--line);
            background: var(--surface);
            backdrop-filter: blur(10px);
            box-shadow: 0 10px 26px rgba(27, 26, 24, 0.08);
            padding: 0.86rem;
            animation: rise .62s ease both;
            display: flex;
            flex-direction: column;
            min-height: 0;
            overflow: hidden;
          }

          .panel-title {
            font-family: "Cormorant Garamond", serif;
            font-size: 1.58rem;
            line-height: 1;
            font-weight: 600;
            color: var(--primary);
          }

          .panel-subtitle {
            color: var(--text-soft);
            font-size: 0.88rem;
            margin-top: 0.14rem;
          }

          .chip-row {
            gap: 0.45rem;
            margin-top: 0.22rem;
            flex-wrap: wrap;
            flex: 0 0 auto;
          }

          .chip-btn {
            border: 1px solid rgba(183, 147, 90, 0.62) !important;
            color: #3e3426 !important;
            background: rgba(245, 236, 220, 0.95) !important;
            border-radius: 999px !important;
            text-transform: none !important;
            font-size: 0.76rem !important;
            letter-spacing: 0.2px !important;
            transition: all 0.2s ease;
          }

          .chip-btn:hover {
            transform: translateY(-1px);
            border-color: rgba(183, 147, 90, 0.9) !important;
            background: rgba(237, 224, 198, 0.95) !important;
            box-shadow: 0 8px 18px rgba(27, 26, 24, 0.09);
          }

          .chat-scroll {
            flex: 1 1 auto;
            min-height: 0;
            border-radius: 14px;
            border: 1px solid rgba(17, 17, 17, 0.14);
            background: rgba(255, 255, 255, 0.92);
            padding: 0.3rem;
          }

          .chat-log {
            width: 100%;
            gap: 0.75rem;
            padding: 0.28rem;
          }

          .chat-row {
            width: 100%;
            display: flex;
            gap: 0.6rem;
            align-items: flex-start;
            animation: rise .24s ease;
          }

          .chat-row.user {
            justify-content: flex-end;
            flex-direction: row-reverse;
          }

          .chat-message-stack {
            width: 100%;
            max-width: 76%;
            gap: 0.3rem;
          }

          .chat-row.user .chat-message-stack {
            align-items: flex-end;
          }

          .chat-avatar {
            margin-top: 0.15rem;
            font-size: 1.03rem;
            color: var(--accent);
          }

          .chat-bubble {
            max-width: 100%;
            border-radius: 14px;
            padding: 0.74rem 0.88rem;
            border: 1px solid rgba(27, 26, 24, 0.14);
            line-height: 1.55;
            font-size: 0.95rem;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
          }

          .chat-row.user .chat-bubble {
            background: linear-gradient(130deg, #23201d, #36312c);
            border-color: rgba(183, 147, 90, 0.62);
            color: #faf7f2;
          }

          .chat-row.assistant .chat-bubble {
            background: rgba(255, 255, 255, 1);
            color: var(--text-main);
          }

          .compose-row {
            margin-top: 0.46rem;
            gap: 0.55rem;
            align-items: stretch;
            width: 100%;
            flex: 0 0 auto;
          }

          .compose-input {
            flex: 1;
          }

          .compose-input textarea {
            min-height: 56px !important;
            max-height: 96px !important;
            color: var(--text-main) !important;
          }

          .compose-input .q-field__control {
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.95) !important;
            border: 1px solid rgba(17, 17, 17, 0.16) !important;
          }

          .compose-input textarea::placeholder {
            color: #7a746e;
          }

          .send-btn {
            min-width: 124px !important;
            border-radius: 12px !important;
            text-transform: none !important;
            font-weight: 600 !important;
            background: linear-gradient(130deg, #23201d, #3a342d) !important;
            border: 1px solid rgba(183, 147, 90, 0.92) !important;
            color: #faf7f2 !important;
            transition: transform .18s ease, box-shadow .18s ease;
          }

          .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(17, 17, 17, 0.15);
          }

          .status-row {
            margin-top: 0.24rem;
            color: var(--text-soft);
            font-size: 0.79rem;
            gap: 0.5rem;
            align-items: center;
            flex: 0 0 auto;
          }

          .recommendation-panel {
            min-height: 0;
          }

          .carousel-wrap {
            margin-top: 0.35rem;
            gap: 0.66rem;
            min-height: 0;
            flex: 1 1 auto;
            overflow: hidden;
          }

          .carousel-controls {
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            flex-wrap: wrap;
          }

          .carousel-nav {
            gap: 0.45rem;
          }

          .carousel-btn {
            border-radius: 10px !important;
            text-transform: none !important;
            font-size: 0.8rem !important;
            color: #faf7f2 !important;
            border: 1px solid rgba(183, 147, 90, 0.82) !important;
            background: linear-gradient(130deg, #23201d, #3a342d) !important;
          }

          .carousel-indicator {
            color: var(--text-soft);
            font-size: 0.82rem;
            letter-spacing: 0.2px;
          }

          .carousel-pills {
            gap: 0.38rem;
            align-items: center;
          }

          .carousel-pill {
            min-width: 34px !important;
            border-radius: 999px !important;
            border: 1px solid rgba(17, 17, 17, 0.18) !important;
            background: rgba(255, 255, 255, 0.95) !important;
            color: var(--primary) !important;
            text-transform: none !important;
          }

          .carousel-pill.active {
            border-color: rgba(183, 147, 90, 0.95) !important;
            background: rgba(183, 147, 90, 0.26) !important;
            color: #111111 !important;
          }

          .broker-row {
            margin-top: 0.38rem;
            justify-content: flex-end;
            flex: 0 0 auto;
          }

          .broker-btn {
            border-radius: 10px !important;
            text-transform: none !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            background: linear-gradient(130deg, #1f7a47, #2f9a5f) !important;
            border: 1px solid rgba(23, 98, 57, 0.95) !important;
            color: #f3fff8 !important;
            padding: 0.22rem 0.76rem !important;
          }

          .broker-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(31, 122, 71, 0.3);
          }

          .carousel-stage {
            width: 100%;
            flex: 1 1 auto;
            min-height: 0;
            overflow: auto;
            padding-right: 0.15rem;
          }

          .prop-card {
            width: 100%;
            min-height: 100%;
            background: linear-gradient(155deg, rgba(255, 255, 255, 0.99), rgba(247, 242, 232, 0.96));
            border: 1px solid rgba(17, 17, 17, 0.15);
            border-radius: 16px;
            padding: 0.72rem;
            transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
          }

          .prop-card:hover {
            transform: translateY(-2px);
            border-color: rgba(183, 147, 90, 0.8);
            box-shadow: 0 14px 26px rgba(17, 17, 17, 0.09);
          }

          .prop-badge {
            display: inline-flex;
            width: fit-content;
            border-radius: 999px;
            border: 1px solid rgba(183, 147, 90, 0.84);
            background: rgba(183, 147, 90, 0.17);
            color: #5f4a21;
            padding: 0.2rem 0.62rem;
            font-size: 0.72rem;
            margin-bottom: 0.42rem;
            letter-spacing: 0.2px;
          }

          .prop-image {
            width: 100%;
            height: 140px;
            border-radius: 12px;
            object-fit: cover;
            border: 1px solid rgba(17, 17, 17, 0.16);
          }

          .prop-placeholder {
            width: 100%;
            height: 140px;
            border-radius: 12px;
            border: 1px dashed rgba(17, 17, 17, 0.28);
            background:
              radial-gradient(circle at 20% 25%, rgba(183, 147, 90, 0.2), transparent 35%),
              linear-gradient(130deg, rgba(255, 255, 255, 0.92), rgba(245, 238, 227, 0.86));
            color: #4b4b52;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
            flex-direction: column;
            font-size: 0.86rem;
          }

          .prop-title {
            margin-top: 0.46rem;
            font-size: 0.98rem;
            font-weight: 600;
            color: var(--primary);
            line-height: 1.28;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }

          .prop-meta {
            margin-top: 0.2rem;
            font-size: 0.79rem;
            color: var(--secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }

          .prop-desc {
            margin-top: 0.24rem;
            font-size: 0.8rem;
            color: #4d4841;
            line-height: 1.33;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
          }

          .prop-more {
            margin-top: 0.5rem;
            border-radius: 10px;
            border: 1px solid rgba(17, 17, 17, 0.16);
            background: rgba(250, 247, 242, 0.9);
            overflow: hidden;
          }

          .prop-more-body {
            margin-top: 0.3rem;
            color: var(--text-soft);
            font-size: 0.82rem;
            line-height: 1.36;
          }

          .chat-scroll .q-scrollarea__thumb,
          .carousel-stage::-webkit-scrollbar-thumb,
          body::-webkit-scrollbar-thumb {
            background: rgba(183, 147, 90, 0.85);
            border-radius: 10px;
          }

          .chat-scroll .q-scrollarea__bar,
          body::-webkit-scrollbar,
          .carousel-stage::-webkit-scrollbar {
            width: 8px;
            height: 8px;
          }

          .chat-scroll .q-scrollarea__container {
            scrollbar-color: rgba(183, 147, 90, 0.85) rgba(17, 17, 17, 0.06);
          }

          .hidden {
            display: none !important;
          }

          @keyframes rise {
            from {
              opacity: 0;
              transform: translateY(11px);
            }
            to {
              opacity: 1;
              transform: translateY(0);
            }
          }

          @keyframes sweep {
            from { transform: translateX(-120%); }
            to { transform: translateX(120%); }
          }

          @media (max-width: 1100px) {
            html,
            body {
              overflow: auto;
            }

            .nicegui-content {
              height: auto;
              overflow: visible;
            }

            .app-shell {
              height: auto;
              min-height: 100vh;
              padding: 1rem 0;
              overflow: visible;
            }

            .main-grid {
              grid-template-columns: 1fr;
            }

            .chat-scroll {
              min-height: 52vh;
            }

            .recommendation-panel {
              min-height: auto;
            }

            .chat-message-stack {
              max-width: 100%;
            }

            .prop-image,
            .prop-placeholder {
              height: 152px;
            }

            .carousel-controls {
              gap: 0.4rem;
            }
          }
        </style>
        """
    )
    ui.add_head_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap" rel="stylesheet">
        <style>
          :root {
            --bg: #F4F7FB; --surface: #FFFFFF; --surface-muted: #EEF2F8; --text-strong: #0F172A;
            --text: #1E293B; --text-soft: #64748B; --line: #D6DEEA; --primary: #1D4ED8;
            --primary-hover: #1E40AF; --accent: #0F766E; --success: #15803D;
          }
          body {
            font-family: "Manrope", sans-serif !important;
            color: var(--text) !important;
            background:
              radial-gradient(circle at 8% 10%, rgba(29,78,216,.18), transparent 40%),
              radial-gradient(circle at 90% 12%, rgba(15,118,110,.14), transparent 42%),
              linear-gradient(150deg, #F9FBFF, var(--bg), #EDF2FA) !important;
            background-size: 130% 130% !important;
            animation: bgShift 16s ease-in-out infinite !important;
          }
          .app-shell { width: min(1380px, 95vw) !important; padding: 1.1rem 0 1.3rem !important; gap: .9rem !important; }
          .hero-shell {
            border-radius: 18px !important; padding: 1rem 1.2rem !important;
            background: linear-gradient(120deg, rgba(15,23,42,.97), rgba(30,64,175,.92), rgba(15,118,110,.85)) !important;
            box-shadow: 0 18px 42px rgba(15,23,42,.22) !important;
          }
          .hero-compact { border-radius: 14px !important; padding: .72rem .9rem !important; }
          .hero-compact .hero-title { font-size: clamp(1.12rem, 1.45vw, 1.45rem) !important; }
          .hero-compact .hero-subtitle { margin-top: .14rem !important; font-size: .84rem !important; }
          .hero-title { font-family: "Sora", sans-serif !important; color: #F8FAFC !important; font-size: clamp(1.45rem, 2.1vw, 2rem) !important; }
          .hero-subtitle { color: #DBEAFE !important; font-size: .95rem !important; }
          .main-grid { gap: .95rem !important; grid-template-columns: 1.33fr .95fr !important; align-items: stretch !important; }
          .right-stack { display: flex; flex-direction: column; gap: .68rem; min-height: 0; height: 100% !important; }
          .conversation-panel { height: 100% !important; min-height: 100% !important; }
          .recommendation-panel { flex: 1 1 auto; min-height: 0; }
          .panel {
            border: 1px solid var(--line) !important; border-radius: 18px !important; background: var(--surface) !important;
            box-shadow: 0 14px 34px rgba(15,23,42,.08) !important; padding: 1rem !important;
          }
          .panel-title { font-family: "Sora", sans-serif !important; font-size: 1.2rem !important; color: var(--text-strong) !important; }
          .panel-subtitle { color: var(--text-soft) !important; font-size: .9rem !important; }
          .chip-btn {
            border: 1px solid #CBD8EC !important; color: #1E3A8A !important; background: #F8FAFF !important;
            font-size: .79rem !important; text-transform: none !important; border-radius: 999px !important;
          }
          .chat-scroll { border: 1px solid #C8D8EF !important; background: #F0F5FF !important; margin-top: .68rem !important; box-shadow: inset 0 0 0 1px rgba(255,255,255,.8) !important; }
          .chat-row { animation: msgIn .3s ease both !important; }
          .chat-bubble { border: 1px solid #C9D9EF !important; font-size: .98rem !important; line-height: 1.6 !important; box-shadow: 0 6px 16px rgba(15,23,42,.06) !important; }
          .chat-row.assistant .chat-bubble { background: #FFFFFF !important; color: var(--text) !important; border-color: #C2D3EC !important; }
          .chat-row.user .chat-bubble { background: linear-gradient(135deg, var(--primary), #1E3A8A) !important; color: #F8FAFC !important; border-color: rgba(30,64,175,.75) !important; box-shadow: 0 8px 18px rgba(29,78,216,.2) !important; }
          .compose-input .q-field__control { border: 1px solid #BFD1EA !important; background: #FFFFFF !important; box-shadow: 0 4px 12px rgba(15,23,42,.07) !important; }
          .compose-input textarea { color: #0F172A !important; font-weight: 500 !important; }
          .send-btn {
            background: linear-gradient(140deg, var(--primary), var(--primary-hover)) !important; border-color: rgba(29,78,216,.95) !important;
            color: #F8FAFC !important; font-weight: 600 !important; text-transform: none !important;
          }
          .carousel-btn { border: 1px solid #C7D6EE !important; background: #F8FAFF !important; color: #1E3A8A !important; font-weight: 600 !important; }
          .carousel-pill { border: 1px solid #D3DDEE !important; background: #FFFFFF !important; color: #1E293B !important; }
          .carousel-pill.active { border-color: var(--primary) !important; background: rgba(29,78,216,.12) !important; color: var(--primary-hover) !important; }
          .prop-card { border: 1px solid #D5DFEF !important; background: linear-gradient(170deg, #FFFFFF, #F7FAFF) !important; border-radius: 16px !important; }
          .prop-image, .prop-placeholder { height: 164px !important; }
          .prop-placeholder {
            border: 1px dashed #AAC0E4 !important;
            background: radial-gradient(circle at 26% 20%, rgba(29,78,216,.15), transparent 42%), linear-gradient(145deg, #F7FAFF, #EEF4FC) !important;
            color: #42566F !important;
          }
          .prop-title { font-family: "Sora", sans-serif !important; color: var(--text-strong) !important; font-size: 1.02rem !important; }
          .prop-desc { color: #475569 !important; font-size: .86rem !important; line-height: 1.45 !important; }
          .prop-more { border: 1px solid #D6E0EF !important; background: #F8FBFF !important; }
          .contact-agent-btn {
            text-transform: none !important; border-radius: 11px !important; font-weight: 600 !important;
            background: linear-gradient(135deg, var(--accent), #0D9488) !important; color: #F0FDFA !important;
            border: 1px solid rgba(15,118,110,.9) !important;
          }
          .contact-agent-btn[disabled] { background: #CBD5E1 !important; border-color: #CBD5E1 !important; color: #6B7280 !important; }
          .schedule-dialog-card { width: min(560px, 92vw) !important; border-radius: 16px !important; border: 1px solid var(--line) !important; }
          .schedule-title { font-family: "Sora", sans-serif !important; color: var(--text-strong) !important; font-size: 1.07rem !important; }
          .schedule-subtitle { color: var(--text-soft) !important; font-size: .88rem !important; }
          .schedule-context { border: 1px solid #D1DEEF !important; border-radius: 11px !important; background: #F7FAFF !important; color: #1E3A8A !important; padding: .58rem .66rem !important; font-size: .85rem !important; }
          .schedule-field { margin-top: .48rem !important; }
          .schedule-grid { margin-top: .48rem !important; display: grid !important; grid-template-columns: 1fr 1fr; gap: .55rem; }
          .field-error { margin-top: .2rem !important; min-height: 1em; color: #B91C1C !important; font-size: .78rem !important; }
          .schedule-actions { margin-top: .72rem !important; display: flex; justify-content: flex-end; gap: .45rem; }
          .cancel-btn { text-transform: none !important; border-radius: 10px !important; border: 1px solid #C9D6EA !important; background: #FFFFFF !important; color: #334155 !important; }
          .confirm-btn { text-transform: none !important; border-radius: 10px !important; border: 1px solid rgba(21,128,61,.9) !important; background: linear-gradient(135deg, var(--success), #16A34A) !important; color: #F0FDF4 !important; }
          .prop-meta-grid { margin-top: .45rem; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .36rem; }
          .prop-meta-item { display: flex; align-items: center; gap: .34rem; border: 1px solid #D9E3F2; background: #F8FAFF; border-radius: 10px; padding: .34rem .42rem; min-width: 0; }
          .prop-meta-icon { color: var(--primary); font-size: .95rem; }
          .prop-meta-text { color: #334155; font-size: .79rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
          .prop-actions { margin-top: auto; padding-top: .66rem; display: flex; justify-content: flex-end; }
          @keyframes bgShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
          @keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
          @media (max-width: 1080px) { .main-grid { grid-template-columns: 1fr !important; } .right-stack { gap: .6rem !important; } .chat-message-stack { max-width: 100% !important; } .chat-scroll { min-height: 52vh !important; } .prop-image, .prop-placeholder { height: 172px !important; } }
          @media (max-width: 640px) { .compose-row { flex-direction: column; } .send-btn { width: 100% !important; } .schedule-grid { grid-template-columns: 1fr !important; } }
          @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
        </style>
        """
    )


def build_ui() -> None:
    _inject_styles()
    history: List[Dict[str, str]] = []
    carousel_state: Dict[str, Any] = {"properties": [], "index": 0}
    scheduler_state: Dict[str, Any] = {
        "selected_property_context": None,
        "selected_date": "",
        "selected_time_slot": "",
    }
    carousel_pills: List[Any] = []
    today = datetime.now().date()
    max_date = today + timedelta(days=14)
    today_str = today.isoformat()
    max_date_str = max_date.isoformat()
    time_slots = [f"{minute // 60:02d}:{minute % 60:02d}" for minute in range(9 * 60, 18 * 60 + 31, 30)]

    def scroll_chat_to_bottom() -> None:
        ui.run_javascript(
            """
            setTimeout(() => {
              const target = document.querySelector('.chat-scroll .q-scrollarea__container');
              if (target) {
                target.scrollTop = target.scrollHeight;
              }
            }, 35);
            """
        )

    def add_chat_message(role: str, text: str, *, auto_scroll: bool = True) -> None:
        row_class = "user" if role == "user" else "assistant"
        icon_name = "person" if role == "user" else "smart_toy"
        with chat_log:
            with ui.row().classes(f"chat-row {row_class}"):
                ui.icon(icon_name).classes("chat-avatar")
                with ui.column().classes("chat-message-stack"):
                    ui.markdown(text).classes("chat-bubble")
        if auto_scroll:
            scroll_chat_to_bottom()

    def _is_valid_email(value: str) -> bool:
        if value.count("@") != 1:
            return False
        local, domain = value.split("@", 1)
        return bool(local) and "." in domain and not domain.startswith(".") and not domain.endswith(".")

    def clear_schedule_errors() -> None:
        name_error.set_text("")
        email_error.set_text("")
        date_error.set_text("")
        time_error.set_text("")

    def reset_schedule_form() -> None:
        schedule_name.value = ""
        schedule_email.value = ""
        schedule_date.value = ""
        schedule_time.value = None
        scheduler_state["selected_date"] = ""
        scheduler_state["selected_time_slot"] = ""
        clear_schedule_errors()

    def open_schedule_dialog() -> None:
        context = scheduler_state.get("selected_property_context")
        if not context:
            ui.notify("Select a recommended property first.", color="warning", timeout=2100)
            return
        context_title = str(context.get("title") or "Selected property")
        context_listing_id = str(context.get("listing_id") or "Not available")
        schedule_context.set_text(f"Property: {context_title} | Listing ID: {context_listing_id}")
        reset_schedule_form()
        schedule_dialog.open()

    def submit_schedule_request() -> None:
        clear_schedule_errors()
        has_error = False

        name = str(schedule_name.value or "").strip()
        email = str(schedule_email.value or "").strip()
        meeting_date = str(schedule_date.value or "").strip()
        meeting_time = str(schedule_time.value or "").strip()

        if not name:
            name_error.set_text("Name is required.")
            has_error = True

        if not email:
            email_error.set_text("Email is required.")
            has_error = True
        elif not _is_valid_email(email):
            email_error.set_text("Enter a valid email address.")
            has_error = True

        if not meeting_date:
            date_error.set_text("Select a date.")
            has_error = True
        else:
            try:
                selected_date = datetime.strptime(meeting_date, "%Y-%m-%d").date()
                if not (today <= selected_date <= max_date):
                    date_error.set_text("Date must be within the next 14 days.")
                    has_error = True
            except ValueError:
                date_error.set_text("Use a valid date.")
                has_error = True

        if not meeting_time:
            time_error.set_text("Select a time slot.")
            has_error = True
        elif meeting_time not in time_slots:
            time_error.set_text("Choose one of the 30-minute slots.")
            has_error = True

        if has_error:
            return

        scheduler_state["selected_date"] = meeting_date
        scheduler_state["selected_time_slot"] = meeting_time
        context = scheduler_state.get("selected_property_context") or {}
        title = str(context.get("title") or "Selected property")
        ui.notify(
            f"Meeting request received for {title} on {meeting_date} at {meeting_time}. "
            f"Our agent will contact you at {email}.",
            color="positive",
            timeout=3400,
        )
        schedule_dialog.close()
        reset_schedule_form()

    def _update_carousel_controls() -> None:
        total = len(carousel_state["properties"])
        index = carousel_state["index"]

        indicator.set_text(f"{index + 1}/{total}" if total else "0/0")
        if total <= 1:
            prev_button.disable()
            next_button.disable()
        else:
            if index <= 0:
                prev_button.disable()
            else:
                prev_button.enable()

            if index >= total - 1:
                next_button.disable()
            else:
                next_button.enable()

        for pill_idx, pill in enumerate(carousel_pills):
            if pill_idx < total:
                pill.set_visibility(True)
                pill.classes(remove="active")
                if pill_idx == index:
                    pill.classes(add="active")
            else:
                pill.set_visibility(False)
                pill.classes(remove="active")

    def _set_carousel_placeholder() -> None:
        badge.set_text("Recommendation")
        card_image.classes(add="hidden")
        card_placeholder.classes(remove="hidden")
        card_title.set_text("Recommendation will appear here")
        meta_price.set_text("Price")
        meta_beds.set_text("Bedrooms")
        meta_handover.set_text("Handover")
        meta_area.set_text("Area")
        card_desc.set_text("Share your targets to receive investor-focused matches.")
        card_more.set_visibility(False)
        scheduler_state["selected_property_context"] = None
        contact_agent_button.disable()

    def _render_carousel() -> None:
        properties = carousel_state["properties"]
        total = len(properties)

        if total == 0:
            carousel_state["index"] = 0
            _set_carousel_placeholder()
            _update_carousel_controls()
            return

        if carousel_state["index"] >= total:
            carousel_state["index"] = total - 1

        index = carousel_state["index"]
        prop = properties[index]
        image_url = str(prop.get("image_url") or "").strip()
        description_full = str(prop.get("description") or "No description available.").strip()
        description_short = _clip_text(description_full, max_len=170)

        badge.set_text(f"Recommendation {index + 1}")
        if image_url:
            card_image.set_source(image_url)
            card_image.classes(remove="hidden")
            card_placeholder.classes(add="hidden")
        else:
            card_image.classes(add="hidden")
            card_placeholder.classes(remove="hidden")

        bedrooms = prop.get("bedrooms", "N/A")
        handover = prop.get("handover", "N/A")
        area = str(prop.get("area") or "Area not specified")
        title = str(prop.get("title") or "Untitled property")
        listing_id = str(prop.get("listing_id") or "N/A")

        card_title.set_text(title)
        meta_price.set_text(_safe_price(prop.get("price_aed")))
        meta_beds.set_text(f"{bedrooms} Beds")
        meta_handover.set_text(str(handover))
        meta_area.set_text(area)
        card_desc.set_text(description_short)
        scheduler_state["selected_property_context"] = {"title": title, "listing_id": listing_id}
        contact_agent_button.enable()

        if description_full and description_full != description_short:
            card_more_body.set_text(description_full)
            card_more.set_visibility(True)
        else:
            card_more.set_visibility(False)

        _update_carousel_controls()

    def set_carousel_properties(properties: List[Dict[str, Any]]) -> None:
        carousel_state["properties"] = properties[:3]
        carousel_state["index"] = 0
        _render_carousel()

    def carousel_prev() -> None:
        if carousel_state["index"] <= 0:
            return
        carousel_state["index"] -= 1
        _render_carousel()

    def carousel_next() -> None:
        total = len(carousel_state["properties"])
        if total == 0 or carousel_state["index"] >= total - 1:
            return
        carousel_state["index"] += 1
        _render_carousel()

    def carousel_select(index: int) -> None:
        total = len(carousel_state["properties"])
        if total == 0 or index >= total:
            return
        carousel_state["index"] = index
        _render_carousel()

    async def process_user_message() -> None:
        message = str(chat_input.value or "").strip()
        if not message:
            return

        chat_input.value = ""
        add_chat_message("user", message)
        send_button.disable()
        chat_input.disable()
        status_row.set_visibility(True)

        try:
            assistant_text, properties = await run_agent_turn(message, history)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": assistant_text})
            add_chat_message("assistant", assistant_text)
            set_carousel_properties(properties)
        except Exception as exc:
            error_text = f"I hit an internal error while processing your request: {exc}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_text})
            add_chat_message("assistant", error_text)
            set_carousel_properties([])
        finally:
            status_row.set_visibility(False)
            send_button.enable()
            chat_input.enable()

    def fill_prompt(prompt: str) -> None:
        chat_input.value = prompt

    with ui.column().classes("app-shell"):
        with ui.row().classes("main-grid"):
            with ui.column().classes("panel conversation-panel"):
                ui.label("Investor Conversation").classes("panel-title")
                ui.label("Describe your budget, area, and objective.").classes("panel-subtitle")

                with ui.row().classes("chip-row"):
                    ui.button(
                        "2.5M AED | 2BR | Downtown",
                        on_click=lambda: fill_prompt("Budget 2.5m AED, 2 bedrooms, Downtown, high rental demand."),
                    ).classes("chip-btn")
                    ui.button(
                        "4M AED | Marina | Capital growth",
                        on_click=lambda: fill_prompt("Budget 4m AED in Dubai Marina focused on capital appreciation."),
                    ).classes("chip-btn")
                    ui.button(
                        "Golden Visa angle",
                        on_click=lambda: fill_prompt("Properties suitable for long-term hold and Golden Visa strategy."),
                    ).classes("chip-btn")

                with ui.scroll_area().classes("chat-scroll"):
                    chat_log = ui.column().classes("chat-log")

                with ui.row().classes("status-row") as status_row:
                    ui.spinner(size="sm", color="warning")
                    ui.label("Analyzing fit across market data, strategy, and legal context.")
                status_row.set_visibility(False)

                with ui.row().classes("compose-row"):
                    chat_input = ui.textarea(
                        placeholder="Example: Budget 3m AED, 2 bedrooms, near metro, handover before 2028."
                    ).props("outlined autogrow").classes("compose-input")
                    send_button = ui.button("Send", icon="north_east", on_click=process_user_message).classes("send-btn")

                add_chat_message(
                    "assistant",
                    "Welcome. Share your investment goals and I will propose high-fit UAE off-plan opportunities with clear reasoning.",
                    auto_scroll=False,
                )

            with ui.column().classes("right-stack"):
                with ui.column().classes("hero-shell hero-compact"):
                    ui.label("Apex Property Advisor").classes("hero-title")
                    ui.label("Professional UAE off-plan guidance for international investors.").classes("hero-subtitle")

                with ui.column().classes("panel recommendation-panel"):
                    ui.label("Recommended Properties").classes("panel-title")
                    ui.label("Top matches update after each message.").classes("panel-subtitle")

                    with ui.column().classes("carousel-wrap"):
                        with ui.row().classes("carousel-controls"):
                            with ui.row().classes("carousel-nav"):
                                prev_button = ui.button("Previous", icon="chevron_left", on_click=carousel_prev).classes(
                                    "carousel-btn"
                                )
                                next_button = ui.button("Next", icon="chevron_right", on_click=carousel_next).classes(
                                    "carousel-btn"
                                )

                            indicator = ui.label("0/0").classes("carousel-indicator")

                            with ui.row().classes("carousel-pills"):
                                for slot in range(3):
                                    pill = ui.button(str(slot + 1), on_click=lambda s=slot: carousel_select(s)).classes(
                                        "carousel-pill"
                                    )
                                    carousel_pills.append(pill)

                        with ui.column().classes("carousel-stage"):
                            with ui.card().classes("prop-card"):
                                badge = ui.label("Recommendation").classes("prop-badge")
                                card_image = ui.image().classes("prop-image hidden")
                                card_placeholder = ui.element("div").classes("prop-placeholder")
                                with card_placeholder:
                                    ui.icon("apartment")
                                    ui.label("Waiting for a property match")

                                card_title = ui.label("").classes("prop-title")
                                with ui.column().classes("prop-meta-grid"):
                                    with ui.row().classes("prop-meta-item"):
                                        ui.icon("payments").classes("prop-meta-icon")
                                        meta_price = ui.label("").classes("prop-meta-text")
                                    with ui.row().classes("prop-meta-item"):
                                        ui.icon("hotel").classes("prop-meta-icon")
                                        meta_beds = ui.label("").classes("prop-meta-text")
                                    with ui.row().classes("prop-meta-item"):
                                        ui.icon("event").classes("prop-meta-icon")
                                        meta_handover = ui.label("").classes("prop-meta-text")
                                    with ui.row().classes("prop-meta-item"):
                                        ui.icon("pin_drop").classes("prop-meta-icon")
                                        meta_area = ui.label("").classes("prop-meta-text")
                                card_desc = ui.label("").classes("prop-desc")

                                with ui.expansion("More details", icon="description").classes("prop-more") as card_more:
                                    card_more_body = ui.label("").classes("prop-more-body")

                                with ui.row().classes("prop-actions"):
                                    contact_agent_button = ui.button(
                                        "Contact Agent", icon="calendar_month", on_click=open_schedule_dialog
                                    ).classes("contact-agent-btn")

    with ui.dialog() as schedule_dialog:
        with ui.card().classes("schedule-dialog-card"):
            ui.label("Schedule a Virtual Meeting").classes("schedule-title")
            ui.label("Choose a time and our agent will contact you to confirm details.").classes("schedule-subtitle")
            schedule_context = ui.label("").classes("schedule-context")

            schedule_name = ui.input("Full Name").props("outlined").classes("schedule-field")
            name_error = ui.label("").classes("field-error")

            schedule_email = ui.input("Email").props("outlined type=email").classes("schedule-field")
            email_error = ui.label("").classes("field-error")

            with ui.row().classes("schedule-grid"):
                with ui.column():
                    schedule_date = ui.input("Preferred Date").props(
                        f"outlined type=date min={today_str} max={max_date_str}"
                    ).classes("schedule-field")
                    date_error = ui.label("").classes("field-error")
                with ui.column():
                    schedule_time = ui.select(options=time_slots, label="Preferred Time").props("outlined").classes(
                        "schedule-field"
                    )
                    time_error = ui.label("").classes("field-error")

            with ui.row().classes("schedule-actions"):
                ui.button("Cancel", on_click=schedule_dialog.close).classes("cancel-btn")
                ui.button("Submit Request", icon="check_circle", on_click=submit_schedule_request).classes("confirm-btn")

    contact_agent_button.disable()
    set_carousel_properties([])


def main() -> None:
    global retriever, agent

    bootstrap_environment()
    validate_properties_db(PROPERTIES_DB_PATH)
    retriever = load_or_build_retriever()
    agent = build_agent()

    build_ui()
    ui.run(title="Apex Property Advisor", reload=False)


if __name__ == "__main__":
    main()
