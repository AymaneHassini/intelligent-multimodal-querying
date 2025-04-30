"""
Basic NL-to-SQL chain implementation.
"""
from langchain.chains import create_sql_query_chain
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from operator import itemgetter

from data.db_connector import get_langchain_db, get_query_tool
from data.schema_utils import table_chain, get_table_details, filter_schema_for_tables
from models.llm import get_langchain_llm
from prompts.templates import create_final_prompt, sql_example_prompt
from utils.sql import clean_sql_query
from chains.answer_chain import format_answer
from data.examples import get_example_selector

def get_basic_chain():
    """
    Create the basic NL-to-SQL chain.
    
    Returns:
        A runnable chain for basic NL-to-SQL conversion
    """
    # 1) Get dependencies
    db = get_langchain_db()
    llm = get_langchain_llm()
    execute_query = get_query_tool()
    full_schema = get_table_details()
    
    # 2) Create few-shot example selector
    example_selector = get_example_selector()
    
    # 3) Create few-shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=sql_example_prompt,
        example_selector=example_selector,
        input_variables=["input", "top_k"],
    )
    
    # 4) Create final prompt
    final_prompt = create_final_prompt(few_shot_prompt)
    
    # 5) SQL query generator
    generate_query = create_sql_query_chain(llm, db, final_prompt)
    
    # 6) Clean query utility
    clean_query = RunnableLambda(clean_sql_query)
    
    # 7) Dynamic table extraction & filtering
    dynamic_chain = (
        RunnablePassthrough.assign(
            selected_tables=lambda x: table_chain.invoke({"question": x["question"]})
        )
        | RunnableLambda(lambda x: (
            print("📂 [DEBUG] Selected tables:", x["selected_tables"]) or x
        ))
        | RunnablePassthrough.assign(
            filtered_schema=lambda x: filter_schema_for_tables(full_schema, x["selected_tables"])
        )
        | RunnableLambda(lambda x: (
            print("📄 [DEBUG] Filtered schema:\n", x["filtered_schema"]) or x
        ))
    )
    
    # 8) Final chain assembly
    chain = (
        dynamic_chain
        | RunnablePassthrough.assign(table_info=lambda x: x["filtered_schema"])

        # Raw SQL generation
        | RunnablePassthrough.assign(
            query=lambda x: (
                (lambda raw_sql: print("🧾 [DEBUG] Raw SQL generated:\n", raw_sql) or raw_sql)(
                    generate_query.invoke({
                        "input": x["input"],
                        "question": x["input"],
                        "table_info": x["table_info"],
                        "messages": x["messages"],
                        "top_k": x["top_k"]
                    })
                )
            )
        )

        # Cleaned SQL before execution
        | RunnablePassthrough.assign(
            cleaned_query=lambda x: (
                print("🧼 [DEBUG] Cleaned SQL:\n", clean_query.invoke(x["query"])) or
                clean_query.invoke(x["query"])
            )
        )

        # Execute query
        | RunnablePassthrough.assign(
            result=lambda x: execute_query.run(x["cleaned_query"])
        )

        # Rephrase final answer
        | RunnableLambda(lambda x: {
            "question": x["input"],
            "query": x["cleaned_query"],
            "result": x["result"],
            "table_info": x["table_info"]
        })
        | RunnableLambda(lambda x: (
            print("🧠 [DEBUG] Rephrasing answer for output:\n", x) or
            format_answer(x["question"], x["query"], x["result"], x["table_info"])
        ))
    )
    
    return chain

def create_chat_history(messages):
    """
    Create a LangChain chat history from message list.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        ChatMessageHistory: LangChain chat history object
    """
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_basic_chain(question, messages):
    """
    Invoke the basic NL-to-SQL chain with a question.
    
    Args:
        question: User's question
        messages: Chat history as a list of message dictionaries
        
    Returns:
        str: Formatted answer to the user's question
    """
    chain = get_basic_chain()
    history = create_chat_history(messages)
    
    # Run the basic NL→SQL chain
    response = chain.invoke({
        "question": question,
        "input": question,
        "top_k": 3,
        "messages": history.messages,
        "constrained_columns": ""
    })
    
    # Record in history
    history.add_user_message(question)
    history.add_ai_message(response)
    
    return response