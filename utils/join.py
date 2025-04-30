"""
SQL JOIN generation utilities.
"""
from typing import List
from langchain_core.output_parsers import StrOutputParser

from models.llm import get_langchain_llm
from prompts.templates import join_prompt

def create_join_chain():
    """
    Create a LangChain for generating SQL JOIN clauses.
    
    Returns:
        A runnable chain that generates SQL JOIN clauses
    """
    llm_join = get_langchain_llm()
    return join_prompt | llm_join | StrOutputParser()

def generate_left_join_query(filtered_schema: str, selected_tables: List[str]) -> str:
    """
    Generate a candidate SQL query by using an LLM to create LEFT JOIN clauses.
    
    Args:
        filtered_schema: Database schema filtered for relevant tables
        selected_tables: List of selected table names
        
    Returns:
        str: Generated SQL query with appropriate joins
    """
    if not selected_tables:
        return ""
        
    # Use the first table as the base table
    base_table = selected_tables[0]
    
    # If only one table, no joins needed
    if len(selected_tables) == 1:
        return f"SELECT * FROM {base_table};"
    
    # Prepare the list of relevant tables as a comma-separated string
    tables_str = ", ".join(selected_tables)
    
    # Create the join chain
    join_chain = create_join_chain()
    
    # Prepare the prompt input
    join_input = {
        "schema": filtered_schema,
        "relevant_tables": tables_str
    }
    
    # Generate join clauses
    join_clauses = join_chain.invoke(join_input)
    
    # Build the final query
    if join_clauses.strip().lower() == "no join":
        return f"SELECT * FROM {base_table};"
    else:
        return f"SELECT * FROM {base_table} {join_clauses};"