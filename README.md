# Intelligent Multimodal Querying System

A unified Natural Language-to-SQL and Advanced Reasoning Chatbot.


## Overview

This project delivers a unified chatbot platform that bridges the gap between everyday language and complex, multimodal database queries. By integrating a schema-aware NL-to-SQL pipeline with a powerful chain-of-thought multimodal reasoning workflow, all within an accessible, bilingual chat interface, it empowers non-technical users to extract meaningful insights from relational databases without writing SQL.

The system features two complementary modes:

1. **Basic NL-to-SQL**: A LangChain-powered pipeline that injects schema metadata into the prompt, uses few-shot examples to guide SQL generation, and applies SQL-cleaning utilities to convert user inputs into valid queries with high precision and low latency.

2. **Advanced Reasoning Pipeline**: A multimodal workflow in which an LLM conducts chain-of-thought reasoning over the full contents of each database row, including text fields, numerical attributes, images, and attached documents, and then employs a fine-tuned BERT classifier to accept, recommend, or reject candidate rows based on insights inferred from the entire row's data.

## Key Features

- **Schema-Aware Query Generation**: Dynamically extracts and filters database schemas to focus the LLM on relevant tables and columns
- **Multimodal Reasoning**: Analyzes text fields, images, and documents within database records
- **BERT-Based Classification**: Fine-tuned model determines which rows truly satisfy complex queries
- **Multilingual Support**: Automatic detection and seamless translation for Arabic users
- **Interactive UI**: Streamlit-based interface with mode selection, chat history, and example queries

## System Architecture

The architecture is organized into these key components:

```
project_root/
│
├── chains/              # Chain implementations
│   ├── answer_chain.py  # Answer formatting
│   ├── basic_chain.py   # Basic NL-to-SQL chain
│   └── advanced_chain.py # Advanced GIQ reasoning
│
├── config/              # Configuration management
│   └── settings.py      # Environment and app settings
│
├── data/                # Data access layer
│   ├── db_connector.py  # Database connection utilities
│   ├── examples.py      # Few-shot learning examples
│   └── schema_utils.py  # Schema extraction utilities
│
├── models/              # Model implementations
│   ├── classifier.py    # BERT classifier for reasoning
│   └── llm.py           # LLM integration (Google Gemini)
│
├── preprocessing/       # Data preprocessing utilities
│   ├── document.py      # Document extraction
│   ├── image.py         # Image preprocessing
│   └── text.py          # Text tokenization
│
├── prompts/             # LLM prompt templates
│   └── templates.py     # Centralized prompt management
│
├── utils/               # Utility functions
│   ├── arabic.py        # Arabic language support
│   ├── join.py          # SQL JOIN generation
│   └── sql.py           # SQL utility functions
│
├── app.py               # Main Streamlit application
├── .env                 # Environment variables (not tracked)
└── README.md            # This file
```

## Technology Stack

- **Python 3.10+** as the implementation language
- **Streamlit** for the interactive chat UI and state management
- **SQLAlchemy and PyMySQL** for database inspection and connections
- **LangChain** for prompt chaining, memory, and SQL tooling
- **Google Gemini 1.5 Pro** for generative, multimodal chain-of-thought reasoning
- **BERT Classifier** (fine-tuned on 3,709 custom examples) for confidence-based row filtering
- **ChromaDB and SemanticSimilarityExampleSelector** for few-shot learning
- **Pillow** for image processing

## Installation and Setup

### Prerequisites

- Python 3.10+
- MySQL database
- Google API key for Gemini API

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AymaneHassini/intelligent-multimodal-querying.git
   cd intelligent-multimodal-querying
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   # Database connection
   db_host=your_db_host
   db_user=your_db_user
   db_password=your_db_password
   db_name=your_db_name
   
   # API keys
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_TRACING_V2=your_langchain_tracing_setting
   
   # Model settings
   CHECKPOINT_PATH=/path/to/bert/checkpoint
   ```

### Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

## Usage

### Basic Mode (NL-to-SQL)

1. Select "Basic (NL-to-SQL)" mode
2. Enter your question in natural language (English or Arabic)
3. The system will translate it to SQL, execute the query, and return the results

![Basic Mode Workflow](https://raw.githubusercontent.com/AymaneHassini/intelligent-multimodal-querying/main/images/figs-ss.pdf)


### Advanced Reasoning Mode

1. Select "Advanced (GIQ)" mode
2. Enter your question in natural language
3. The system will:
   - Identify relevant tables
   - Generate JOIN clauses if needed
   - Process each row with chain-of-thought reasoning
   - Classify rows as relevant/irrelevant with BERT
   - Generate a final answer based on matching rows

![Advanced Mode Workflow](https://raw.githubusercontent.com/AymaneHassini/intelligent-multimodal-querying/main/images/giq-enh-f.pdf)


## Performance

- **Basic mode**: End-to-end latency under 2s
- **Advanced mode**: Per-row processing latency under 2.3s
- **Classification metrics**: Precision: 0.9697, Recall: 0.9708, F1-Score: 0.9739
- **SQL accuracy**: ~88% for single-table queries, ~71% for complex joins

## Future Work

- **On-Premise Deployment**: Deploy compact multimodal LLMs locally for enhanced privacy and reduced latency
- **Additional Data Modalities**: Extend support to audio, video, and sensor data
- **Domain-Specific Fine-Tuning**: Train models on industry-specific data
- **NoSQL Support**: Expand to graph and document databases
- **UX Improvements**: Visual query builders, explanation panels, adaptive UI


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dr. Chakiri Houda for supervision and guidance
- Al Akhawayn University School of Science and Engineering