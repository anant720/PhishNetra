# PhishNetra System Design

## 1. System Architecture Overview
PhishNetra employs a microservices-inspired architecture consisting of a FastAPI-based backend API and a Next.js frontend application. The core scam detection logic is powered by an ensemble of multiple AI/ML models. A critical component is the safe and analytical URL analysis pipeline.

```
+----------------+       +-------------------+       +--------------------+
|  User (Browser)| <---> |  Frontend (Next.js) | <---> |  Backend API (FastAPI) |
+----------------+       +-------------------+       +--------------------+
                                    ^
                                    | HTTP/S
                                    v
                          +-------------------------+
                          |   PhishNetra Service    |
                          | (Decision Fusion, URL Analyzer) |
                          +-------------------------+
                                    ^
                                    |
            +-----------------------+-----------------------+
            |                       |                       |
+-----------+-----------+ +-----------+-----------+ +-----------+-----------+
| FastText Embeddings   | | Sentence Transformers | | DistilBERT Classifier |
| (Keyword/Semantic Sim) | | (Intent/Semantic Patterns)| | (Contextual Analysis) |
+-----------------------+ +-----------+-----------+ +-----------+-----------+
            |                               |
            +-------------------------------+
                                    v
                          +-----------------------+
                          |   FAISS Similarity    |
                          |   (Known Scam Variants) |
                          +-----------------------+
```

## 2. High-Level Architecture Description
The system is divided into three main components:
- **Frontend**: A Next.js application providing the user interface for inputting messages and displaying analysis results. It communicates with the backend API.
- **Backend API**: A FastAPI application that exposes REST endpoints for scam analysis. It orchestrates the AI/ML models, performs preprocessing, and aggregates results.
- **AI/ML Models**: A suite of specialized deep learning models, including FastText for basic semantic similarity, Sentence Transformers for intent detection, DistilBERT for contextual classification, and FAISS for similarity search against known scam patterns. A Decision Fusion component combines their outputs.

## 3. AI/ML Design Overview
PhishNetra utilizes a multi-model ensemble approach to leverage the strengths of different AI architectures:
- **FastText Embeddings**: Captures character-level and word-level information, effective for handling misspellings, slang, and multilingual text (like Hinglish). It contributes to semantic similarity scores.
- **Sentence Transformers (MiniLM)**: Generates sentence embeddings to understand the semantic meaning and intent of messages. It identifies patterns related to financial transactions, phishing indicators, emotional manipulation, etc.
- **DistilBERT Classifier**: A fine-tuned transformer model for high-accuracy contextual classification. It analyzes the message context to classify it as a scam or legitimate, providing attention weights for explainability.
- **FAISS Similarity Search**: A highly efficient similarity search engine. It compares incoming messages against a database of known scam variants and legitimate messages to detect similar threats, even unseen ones.
- **Decision Fusion**: This central component aggregates the predictions (risk scores, confidence) from all individual models using a weighted ensemble approach. It dynamically assigns a primary threat category and generates a comprehensive risk score (0-100). The weights are configurable and can be learned/tuned.

## 4. URL Analysis Approach (High-Level)
A dedicated, safe, and analytical URL analysis pipeline is integrated into the backend. This pipeline operates without executing any untrusted code:
- **Extraction**: Identifies full URLs, shortened links, domain mentions, and QR code textual references.
- **Domain Structure Analysis**: Checks for typosquatting, lookalike domains, excessive subdomains, and unusual TLDs.
- **HTTPS & Certificate Signals**: Verifies HTTPS usage and inspects SSL certificate metadata (issuer, age, validity) without full handshake.
- **Domain Age & Reputation**: Employs heuristics (e.g., keywords, TLDs) and simulated checks for domain age and reputation.
- **Redirection Behavior (Safe Mode)**: Simulates HTTP HEAD requests to detect redirection chains, excessive redirects, and mismatched final domains, without fetching full page content or executing scripts.
- **Content Semantic Fingerprinting**: Fetches only minimal page metadata (e.g., title, header tags, presence of login/payment forms) to detect impersonation or malicious intent. No JavaScript execution or form submission.
- **Risk Integration**: The output of the URL analysis (URL Risk Score, Verdict, Signals, Reasoning) directly contributes to the overall message's risk score and threat categorization within the Decision Fusion component.

## 5. Data Flow
1. **User Input**: A user enters message text (and optionally, URLs) into the Frontend web application.
2. **API Request**: The Frontend sends an asynchronous POST request to the Backend API's `/analyze` endpoint.
3. **Backend Preprocessing**: The Backend's `PhishNetraService` preprocesses the input text, extracting features and identifying URLs.
4. **URL Analysis**: If URLs are found, the `URLAnalyzer` performs its multi-stage, safe analysis.
5. **Model Inference**: The preprocessed text and URL analysis results are fed to the individual AI/ML models (FastText, Sentence Transformers, DistilBERT, FAISS).
6. **Decision Fusion**: The `DecisionFusion` engine collects predictions from all models and the URL analyzer, applying weighted logic to calculate an overall risk score and determine threat categories.
7. **Explainability**: The `ReasoningEngine` generates a human-readable explanation and highlights suspicious phrases based on model outputs.
8. **API Response**: The Backend returns the comprehensive analysis results (risk score, category, explanation, URL analysis) to the Frontend.
9. **Frontend Display**: The Frontend renders these results in a user-friendly interface, including a dedicated URL analysis tab.

## 6. Technology Stack
- **Backend**: Python 3.9+, FastAPI, Uvicorn, Pydantic, httpx, spaCy, nltk, numpy.
- **AI/ML Libraries**: PyTorch, Transformers (for conceptual backing; simplified versions used in hackathon context), sentence-transformers, faiss-cpu, FastText.
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, `@tanstack/react-query`, `axios`, `framer-motion`, `react-hot-toast`, `recharts`.
- **Database**: None explicitly used for this hackathon project (in-memory data for models).
- **Logging**: `structlog`.
- **Development/Deployment**: Docker, docker-compose.

## 7. Testing & Validation Approach
- **Unit Tests**: Individual components (e.g., `URLAnalyzer` methods, `DecisionFusion` logic) are tested for correctness.
- **Integration Tests**: End-to-end API tests to ensure the full analysis pipeline functions correctly.
- **URL-Specific Tests**: Comprehensive test cases for URL analysis, including phishing links, shortened URLs, brand impersonation, and legitimate URLs, with assertions on risk scores and verdicts.
- **Scam Category Validation**: Testing with various scam message examples to ensure correct categorization (Financial, Phishing, Impersonation, etc.).
- **Performance Testing**: Basic timing measurements for API response latency.

## 8. Limitations
- **Model Simplification**: For hackathon purposes, some AI models are represented by simplified implementations or rely on heuristics rather than full-scale deep learning models requiring extensive training data and computational resources.
- **Real-time Data**: Domain age and reputation checks are simulated or based on basic heuristics rather than real-time WHOIS lookups or commercial threat intelligence feeds.
- **Content Fetching Depth**: Content semantic fingerprinting is limited to minimal page metadata (first 10KB) without rendering or JavaScript execution, which might miss sophisticated obfuscation techniques.
- **Language Scope**: Primarily optimized for English and Hinglish; performance on other languages may vary.
- **Scalability (Hackathon)**: While the architecture is designed for scalability, the actual deployment might not include fully optimized scaling groups or load balancing.
- **False Positives/Negatives**: As with any AI system, there's a potential for false positives (legitimate messages flagged as scams) or false negatives (scams missed), especially with highly novel or evasive attacks.
- **No User Management**: No features for user accounts, history, or personalized settings.