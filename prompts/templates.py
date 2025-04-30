"""
Centralized prompt templates for LLM interactions.
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
    PromptTemplate
)
from data.examples import get_example_selector

# SQL Prompt Templates
sql_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}\nSQLQuery:"),
    ("ai", "{query}"),
])

# Answer Prompt Template
answer_prompt = PromptTemplate.from_template(
"""Given the following user question, corresponding SQL query, and SQL result, answer the user question.
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# Arabic Translation Prompt
arabic_prompt = ChatPromptTemplate.from_template(
    "Translate the following answer into Arabic, preserving its meaning and tone:\n\n{answer}"
)

# JOIN Generation Prompt
join_prompt = ChatPromptTemplate.from_template(
"""You are a SQL expert. You are given a database schema (with table names, columns, primary keys, and foreign key relationships) and a list of relevant tables.
Schema:
{schema}
Relevant Tables:
{relevant_tables}
Assume the first table in the list is the base table. Generate appropriate SQL LEFT JOIN clauses to join the base table to each of the other relevant tables using explicit JOIN syntax.
If no join condition can be inferred from the schema, output exactly "No join".
Output only the JOIN clauses.
"""
)

# Per-row Reasoning Guidelines
REASONING_GUIDELINES = """
IMPORTANT GUIDELINES:
1. You will see every field in this record—treat them all as evidence.
2. Walk through each field step by step (chain-of-thought) to decide if this record matches the user's question.
3. Conclusion: [Yes or No]
4. Match Score: [0-100]
5. Use your reasoning and inference/context understanding capabilities.
"""

# Final SQL Generation Prompt
def create_final_prompt(few_shot_prompt):
    """Create the final prompt template with few-shot examples."""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a MySQL expert. Use this schema:
{table_info}
GUIDELINES:
1. Use explicit JOINs
2. Handle NULLs with COALESCE
3. Add LIMIT 100 if missing
4. Use table aliases"""
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ])

def create_row_reasoning_prompt(field_lines, user_prompt):
    """Create a prompt for reasoning about a single database row."""
    return f"""
User's question: "{user_prompt}"
Here is one candidate record with all its fields:
{field_lines}
{REASONING_GUIDELINES}
"""

# Create the few-shot prompt with examples
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=sql_example_prompt,
    example_selector=get_example_selector(),
    input_variables=["input", "top_k"],
)

# Pre-create the final prompt
final_prompt = create_final_prompt(few_shot_prompt)
