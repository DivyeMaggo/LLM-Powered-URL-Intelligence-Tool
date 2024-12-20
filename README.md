# LLM-Powered-URL-Intelligence-Tool

## Overview

The **LLM-Powered-URL-Intelligence-Tool** is a web application that extracts, processes, and answers user queries based on data from up to three user-provided URLs. Using LangChain, the tool scrapes data from these URLs, processes it into chunks, and stores it in FAISS (a similarity search library). User queries are matched against the stored embeddings, and relevant chunks are retrieved. These chunks are then processed using advanced LLMs (Large Language Models) with map-reduce techniques, ensuring the final response is coherent, context-aware, and backed by source references.

## Features

- **URL Data Extraction**: Extracts data from up to three user-provided URLs.
- **Data Processing**: Data is chunked, embedded, and stored for similarity search using FAISS.
- **Query Processing**: User queries are answered by retrieving the most relevant chunks from the stored embeddings and processed using LLMs.
- **Context-Aware Responses**: The final output is a coherent, accurate answer, with references to the source URLs for verification.
- **User-Friendly Web Interface**: Built using Streamlit for an intuitive and responsive UI.

## Technologies Used

- **Python**: Core programming language for backend logic and processing.
- **LangChain**: A framework for building LLM-powered applications, used for data extraction, processing, and interaction with the LLMs.
- **FAISS**: A library for efficient similarity search and clustering of embeddings.
- **Streamlit**: A web framework for creating the front-end interface.
- **Prompt Engineering**: Advanced techniques used to interact with the LLMs for efficient and accurate responses.
- **OpenAI API / Hugging Face**: For the LLMs processing and answering queries.

## Installation

### Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.7+
- pip (Python package installer)

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/LLM-Powered-URL-Intelligence-Tool.git
cd LLM-Powered-URL-Intelligence-Tool
streamlit run main.py
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Step 3: Set up the API keys
To interact with the LLMs and other services, you need to provide API keys. Make a '.env' file in the parent directory and set the following environment variables in:
```bash
OPENAI_API_KEY='your-openai-api-key'
```

### Step 4: Step 4: Run the app
Start the Streamlit application:
```bash
streamlit run app.py
```

## Usage
- Input URLs: Provide up to three URLs to extract data from. The tool will scrape the content from these URLs and prepare it for query answering.
- Ask a Query: Enter any question related to the content from the provided URLs.
- Get Answer: The application will process the query, retrieve relevant information, and display the answer along with source URL references.
