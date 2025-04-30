"""
Examples module for few-shot learning in NL-to-SQL.
"""
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Example data for few-shot learning
examples = [
  {
    "input": "List all married citizens born after 1995",
    "query": "SELECT NOM, PRENOM, DATENAISSFR FROM citoyen WHERE MARIAGE = 1 AND SUBSTRING(DATENAISSFR,1,4) > '1995';"
  },
  {
    "input": "Show all divorced female citizens",
    "query": "SELECT NOM, PRENOM, CIN FROM citoyen WHERE DIVORCE = 1 AND SEXE = 1;"
  },
  {
    "input": "Find births at Hôpital Ibn Sina with male babies",
    "query": "SELECT nommere, prenommere, datenaissanceenfant FROM avisnaissance WHERE hopital = 'Hôpital Ibn Sina' AND sexeenfant = 0;"
  },
  {
    "input": "Count births per hospital in January 2023",
    "query": "SELECT hopital, COUNT(*) AS total_births FROM avisnaissance WHERE datenaissanceenfant LIKE '2023-01%' GROUP BY hopital;"
  },
  {
    "input": "List marriages from 2005 with declaration dates",
    "query": "SELECT idmariage, datemariage, datedeclamariage FROM mariage WHERE datemariage LIKE '2005-%';"
  },
  {
    "input": "Find citizens both married and divorced",
    "query": "SELECT NOM, PRENOM FROM citoyen WHERE MARIAGE = 1 AND DIVORCE = 1;"
  },
  {
    "input": "Show deaths in Casablanca during 2021",
    "query": "SELECT c.NOM, c.PRENOM, m.DECESDATEFR FROM mentiondeces m JOIN citoyen c ON m.CITOYENID = c.CITOYENID WHERE m.LIEUDECESFR = 'Casablanca' AND m.DECESDATEFR LIKE '2021-%';"
  },
  {
    "input": "Recent name changes in 2020",
    "query": "SELECT c.NOM, mc.TEXTECHGTFR FROM mentionchangement mc JOIN citoyen c ON mc.CITOYENID = c.CITOYENID WHERE mc.DATECHGTFR LIKE '2020-%';"
  },
  {
    "input": "Undeclared males born before 2000",
    "query": "SELECT nom, prenom FROM nondeclare WHERE sexe = 0 AND SUBSTRING(datenaiss,1,4) < '2000';"
  },
  {
    "input": "Find births with invalid hours (24+)",
    "query": "SELECT avisid, heurenaissanceenfant FROM avisnaissance WHERE heurenaissanceenfant >= 24;"
  },
  {
    "input": "Marriages without divorce records",
    "query": "SELECT m.* FROM mariage m LEFT JOIN divorce d ON m.citoyenidepoux = d.citoyenidepoux AND m.citoyenidepouse = d.citoyenidepouse WHERE d.iddivorce IS NULL;"
  },
  {
    "input": "Births by month in 2023",
    "query": "SELECT SUBSTRING(datenaissanceenfant, 6, 2) AS month, COUNT(*) FROM avisnaissance WHERE datenaissanceenfant LIKE '2023-%' GROUP BY month;"
  },
  {
    "input": "Mothers with multiple births",
    "query": "SELECT nommere, prenommere, COUNT(*) AS births FROM avisnaissance GROUP BY nommere, prenommere HAVING COUNT(*) > 1;"
  },
  {
    "input": "Citizens with both birth and death records (assuming mother's names match)",
    "query": "SELECT c.NOM, c.PRENOM, a.datenaissanceenfant, d.DECESDATEFR FROM citoyen c JOIN avisnaissance a ON c.NOM = a.nommere AND c.PRENOM = a.prenommere JOIN mentiondeces d ON c.CITOYENID = d.CITOYENID;"
  },
  {
    "input": "Find citizens married in 2020 but not divorced",
    "query": """SELECT c.* FROM citoyen c 
                JOIN mariage m ON c.CITOYENID = m.citoyenidepoux OR c.CITOYENID = m.citoyenidepouse 
                LEFT JOIN divorce d ON m.idmariage = d.idmariage 
                WHERE m.datemariage LIKE '2020-%' AND d.iddivorce IS NULL;"""
  },
  {
    "input": "List citizens with name changes affecting both first and last names",
    "query": """SELECT c.NOM, c.PRENOM, mc.TEXTECHGTFR 
                FROM mentionchangement mc 
                JOIN citoyen c ON mc.CITOYENID = c.CITOYENID 
                WHERE mc.TEXTECHGTFR LIKE '%nom%' AND mc.TEXTECHGTFR LIKE '%prénom%';"""
  },
  {
    "input": "Find babies born between 8 AM and 6 PM",
    "query": "SELECT * FROM avisnaissance WHERE heurenaissanceenfant BETWEEN 8 AND 18;"
  }
]

@st.cache_resource
def get_example_selector():
    """
    Create and return a semantic similarity example selector.
    
    Returns:
        SemanticSimilarityExampleSelector: For selecting relevant examples
    """
    chroma_db_path = "./chroma_db"

    # Ensure directory exists
    os.makedirs(chroma_db_path, exist_ok=True)

    # Define embedding function
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Initialize ChromaDB
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_function,
        persist_directory=chroma_db_path,
    )

    # Create Example Selector with Chroma as vectorstore_cls
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,  # Example data defined above
        embedding_function,
        vectorstore_cls=Chroma,
        k=4,
        input_keys=["input"]
    )

    # Ensure persistence
    vector_store.persist()

    return example_selector