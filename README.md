# Chatbot NLU Enhancement 

This is a simple Streamlit app that produces RASA format NLU based on a provided article.

## Requirements

- Python 3.7.3+

## Installation

1. Install the required Python packages:
   
   ```
   pip install -r requirements.txt
   ```
2. Put the openai.key to the credentials folder
 
   ```
   chatbot-nlu-enhancement/
     ...
     ├── credentials/openai.key
     ├── README.md
     ├── requirements.txt
     ...
   ```

## Usage

1. Run the Streamlit app:
   
   ```python
   streamlit run app.py
   ```
2. (Optional)Put some article text files under the `articles` folder.


3. Open the app in your browser:   
   http://localhost:8501



<h1 align="center">
📖Article2Rasa
</h1>

Accurate answers and instant citations for your documents.

## 🔧 Features

- Upload documents 📁(PDF, DOCX, TXT) and answer questions about them.
- Cite sources📚 for the answers, with excerpts from the text.

## 💻 Running Locally

1. Clone the repository📂

```bash
git clone https://github.com/mmz-001/knowledge_gpt
cd knowledge_gpt
```

2. Install dependencies with [Poetry](https://python-poetry.org/) and activate virtual environment🔨

```bash
poetry install
poetry shell
```

3. Run the Streamlit server🚀

```bash
cd knowledge_gpt
streamlit run main.py
```

## 🚀 Upcoming Features

- Add support for more formats (e.g. webpages 🕸️, PPTX 📊, etc.)
- Highlight relevant phrases in citations 🔦
- Support scanned documents with OCR 📝
- More customization options (e.g. chain type 🔗, chunk size📏, etc.)