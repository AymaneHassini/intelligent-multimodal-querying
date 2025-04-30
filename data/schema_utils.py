"""
Database schema utilities for extracting and processing schema information.
"""
import streamlit as st
from typing import List
from operator import itemgetter
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, LLM_MODEL
from data.db_connector import get_inspector

# Make sure environment variables are not None
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ValueError("Missing one or more DB environment variables in .env (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME).")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) -> List[str]:
    """Extract table names from Pydantic model instances."""
    return [table.name for table in tables]

@st.cache_data
def get_table_details():
    """
    Extract table details from the database and format as a string.
    
    Returns:
        A formatted string containing details of all tables, columns, 
        primary keys, and foreign keys in the database.
    """
    inspector = get_inspector()
    table_names = inspector.get_table_names()
    metadata_str = ""
    
    for table_name in table_names:
        metadata_str += f"Table Name: {table_name}\nColumns:\n"
        
        # Get column details
        columns = inspector.get_columns(table_name)
        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            is_nullable = col["nullable"]
            metadata_str += f" - {col_name} ({col_type}), nullable={is_nullable}\n"
        
        # Get primary key details
        pk = inspector.get_pk_constraint(table_name)
        pk_columns = ", ".join(pk.get("constrained_columns", []))
        metadata_str += f"Primary Key Columns: {pk_columns if pk_columns else 'None'}\n"
        
        # Get foreign key details
        fks = inspector.get_foreign_keys(table_name)
        if fks:
            metadata_str += "Foreign Keys:\n"
            for fk in fks:
                fk_cols = ", ".join(fk.get("constrained_columns", []))
                referred_table = fk.get("referred_table", "Unknown")
                referred_cols = ", ".join(fk.get("referred_columns", []))
                metadata_str += f" - {fk_cols} -> {referred_table}({referred_cols})\n"
        else:
            metadata_str += "Foreign Keys: None\n"
            
        metadata_str += "\n"
        
    return metadata_str

def filter_schema_for_tables(full_schema: str, selected_tables: List[str]) -> str:
    """
    Filter the full schema to include only selected tables.
    
    Args:
        full_schema: The complete database schema as a string
        selected_tables: List of table names to include
        
    Returns:
        Filtered schema containing only the selected tables
    """
    if not selected_tables:
        return full_schema
        
    return "\n\n".join(
        section for section in full_schema.split("\n\n")
        if any(table in section for table in selected_tables)
    )

# Get the database schema once at module load time
table_details = get_table_details()

# Build the prompt for the table-chain including foreign key metadata
table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question.
The tables (along with their columns, primary keys, and foreign keys) are:
{table_details}
Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

# Create the table selection chain
table_chain = (
    {"input": itemgetter("question")}
    | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
    | get_tables
)