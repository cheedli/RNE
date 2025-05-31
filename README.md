``
rne-chatbot/
│
├── app.py                  # Main Flask application
├── config.py               # Configuration settings
├── data/
│   └── rne_laws.json       # Your RNE laws JSON data
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py      # Load and preprocess JSON data
│   └── text_processor.py   # Text preprocessing utilities
│
├── retrieval/
│   ├── __init__.py
│   ├── faiss_retriever.py  # Vector-based retrieval with FAISS
│   ├── bm25_retriever.py   # Keyword-based retrieval with BM25
│   └── hybrid_retriever.py # Combined retrieval system
│
├── llm/
│   ├── __init__.py
│   ├── groq_client.py      # Integration with Groq API
│   └── prompt_templates.py # Prompt engineering for the LLM
│
├── utils/
│   ├── __init__.py
│   ├── language_detector.py # Detect input language
│   └── response_formatter.py # Format responses appropriately
│
├── templates/
│   ├── index.html          # Web interface
│   └── chat.html           # Chat interface
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── chat.js
│
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
``
