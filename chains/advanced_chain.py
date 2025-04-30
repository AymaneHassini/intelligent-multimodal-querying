"""
Advanced integrated pipeline implementation with row-level reasoning.
"""
from tqdm import tqdm
from sqlalchemy import inspect, types
import torch
import torch.nn.functional as F

from data.db_connector import get_langchain_db, get_query_tool
from data.schema_utils import table_chain, get_table_details, filter_schema_for_tables
from models.llm import load_llm
from models.classifier import load_classifier_bert
from preprocessing.image import preprocess_and_check_image
from preprocessing.text import tokenize
from prompts.templates import REASONING_GUIDELINES
from utils.join import generate_left_join_query
from utils.sql import build_where_clause
from chains.answer_chain import format_answer

def process_candidate_row(
    row: tuple,
    columns: list[str],
    image_idx: int = None,
    doc_idx: int = None,
    user_prompt: str = ""
) -> str:
    """
    Process a single candidate row with the LLM.
    
    Args:
        row: Database row as a tuple
        columns: Column names
        image_idx: Index of image column (if any)
        doc_idx: Index of document column (if any)
        user_prompt: Original user question
        
    Returns:
        str: LLM reasoning about the row
    """
    # Get LLM
    llm = load_llm()
    
    # 1) Render all column:value pairs
    field_lines = "\n".join(f"- {columns[i]}: {row[i]!r}" for i in range(len(columns)))
    print("🔍 [DEBUG] Field lines for this row:\n", field_lines)

    # 2) Build chain-of-thought prompt
    prompt = f"""
    User's question: "{user_prompt}"
    Here is one candidate record with all its fields:
    {field_lines}
    {REASONING_GUIDELINES}
    """
    print("📝 [DEBUG] Full prompt sent to LLM:\n", prompt)

    # 3) Dispatch to correct modality
    if image_idx is not None and row[image_idx] is not None:
        # Preprocess the image
        image = preprocess_and_check_image(row[image_idx])
        # Feed the image and prompt to the LLM
        resp = llm.generate_content([image, "\n\n", prompt])
    elif doc_idx is not None and row[doc_idx] is not None:
        # Feed the document and prompt to the LLM
        document = row[doc_idx]
        resp = llm.generate_content([document, prompt])
    else:
        # Text-only prompt
        resp = llm.generate_content([prompt])
        
    print("📬 [DEBUG] LLM response:", resp.text)
    return resp.text

def invoke_advanced_chain(user_query: str, messages: list[dict]) -> str:
    """
    Invoke the advanced integrated pipeline.
    
    Args:
        user_query: User's question
        messages: Chat history
        
    Returns:
        str: Formatted answer
    """
    # 1) Schema & table selection
    full_schema = get_table_details()
    selected_tables = table_chain.invoke({
        "question": user_query,
        "input": user_query,
        "constrained_columns": ""
    })
    print("DEBUG ▶ Selected tables:", selected_tables)
    
    if not selected_tables:
        return "No relevant table found for your query."
        
    filtered_schema = filter_schema_for_tables(full_schema, selected_tables)
    print("DEBUG ▶ Filtered schema:\n", filtered_schema)

    # 2) Build candidate_sql
    candidate_sql = generate_left_join_query(filtered_schema, selected_tables)
    print("DEBUG ▶ Advanced candidate_sql:", candidate_sql)

    # 3) Execute & fetch
    db = get_langchain_db()
    tool = get_query_tool()
    
    # Warm up the database connection
    tool.run(candidate_sql)
    
    # Get raw connection and cursor
    conn = db._engine.raw_connection()
    cursor = conn.cursor()
    cursor.execute(candidate_sql)
    rows = cursor.fetchall()
    columns = [d[0] for d in cursor.description]
    print("DEBUG ▶ Columns:", columns, "| #rows:", len(rows))

    # 4) Dynamic detection of image/doc columns
    insp = inspect(db._engine)
    cols_info = insp.get_columns(selected_tables[0])
    sample_row = rows[0] if rows else None

    image_idx = None
    doc_idx = None
    
    for i, col_name in enumerate(columns):
        info = next((c for c in cols_info if c["name"] == col_name), None)
        if not info:
            continue
        col_type = info["type"]

        # 1) binary BLOBs => could be image or doc, disambiguate by name
        if isinstance(col_type, types.LargeBinary):
            if any(tok in col_name.lower() for tok in ("img", "photo", "picture")):
                image_idx = i
            else:
                doc_idx = i

        # 2) string columns: sniff extensions on the first row's value
        elif isinstance(col_type, (types.String, types.Text, types.VARCHAR)) and sample_row:
            val = sample_row[i]
            if isinstance(val, str):
                low = val.lower()
                if low.endswith((".jpg", ".jpeg", ".png")):
                    image_idx = i
                elif low.endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx")):
                    doc_idx = i

    print(f"DEBUG ▶ image_idx={image_idx}, doc_idx={doc_idx}")

    # 5) Primary key detection
    pk_info = insp.get_pk_constraint(selected_tables[0])
    pk_col = (pk_info.get("constrained_columns") or [columns[0]])[0]
    print(f"DEBUG ▶ Primary key column: {pk_col}")

    # 6) Per-row CoT + BERT classification
    trainer, tokenizer = load_classifier_bert()
    accepted = []
    
    for row in tqdm(rows, desc="Processing candidate rows"):
        # Get LLM reasoning for this row
        llm_out = process_candidate_row(row, columns, image_idx, doc_idx, user_query)
        print("🔖 [DEBUG] LLM output:", llm_out)
        
        # Prepare input for BERT
        stripped = llm_out.replace("\n", " ").replace("*", "")
        bert_in = f"Question: {user_query} Answer: {stripped}"
        print("📝 [DEBUG] BERT input:", bert_in)
        
        # Tokenize and classify
        ds = tokenize(bert_in, tokenizer)
        pred = trainer.predict(ds)
        
        # Get probabilities and prediction
        probs = F.softmax(torch.tensor(pred.predictions), dim=-1)
        cls = int(probs.argmax().item())
        print("📊 [DEBUG] BERT probs:", probs.tolist(), "→ class", cls)
        
        # If classified as positive, add to accepted rows
        if cls == 0:
            idx = columns.index(pk_col)
            accepted.append(row[idx])

    if not accepted:
        return "None of the candidate rows passed the advanced reasoning filter."

    # 7) Final filtered SQL on base table
    where = build_where_clause(accepted, pk_col)
    final_sql = f"SELECT * FROM {selected_tables[0]} WHERE {where};"
    print("DEBUG ▶ Final SQL:", final_sql)
    
    # Execute final query
    final_rows = tool.run(final_sql)
    final_str = "\n".join(map(str, final_rows))

    # 8) Friendly answer
    final_answer = format_answer(user_query, final_sql, final_str, filtered_schema)

    return final_answer