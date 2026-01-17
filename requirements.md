# PhishNetra Project Requirements

## 1. Project Overview
PhishNetra is a hackathon-level AI-based scam and phishing detection web application. The system is designed to analyze message text and optional URLs to identify potential scams, providing a risk score, scam category, and a brief explanation of the detection. The backend, frontend, and AI/ML models are already implemented.

## 2. Problem Statement
The proliferation of scam and phishing attempts through various digital communication channels poses a significant threat to individuals and organizations, leading to financial fraud, identity theft, and other malicious activities. Traditional rule-based detection systems often struggle to keep up with evolving scam tactics, necessitating a more adaptive and intelligent solution.

## 3. Objectives
- To develop an AI-powered system capable of detecting various types of scam and phishing messages.
- To provide a clear risk score (0-100) indicating the severity of a detected scam.
- To categorize identified scams into distinct types such as Phishing, Impersonation, Financial Scam, Employment Scam, Investment Fraud, etc.
- To offer a concise, human-readable explanation for the detected risk and categorization.
- To create a user-friendly web interface for submitting messages and viewing analysis results.
- To implement safe and analytical URL analysis to detect malicious links without execution.

## 4. Scope of the Project
The project focuses on building a functional web application for scam detection using multiple deep learning architectures.
- **Input**: Message text (mandatory) and optional URLs within the message.
- **Core Logic**: AI/ML-based analysis using FastText, Sentence Transformers, DistilBERT, and FAISS similarity search, fused by a decision engine.
- **Output**: Risk score (0-100), scam category, and a short explanation.
- **User Interface**: A web-based frontend for interaction.
- **URL Analysis**: A safe, read-only pipeline for analyzing URLs present in messages.

The scope explicitly EXCLUDES:
- User authentication and authorization features.
- Persistent storage of analysis results (beyond immediate display).
- Advanced user management.
- Real-time threat intelligence feeds beyond simulated data.
- Extensive hyperparameter tuning or model retraining pipelines within the application itself.

## 5. Functional Requirements
- The system MUST accept message text as input for analysis.
- The system MUST extract and analyze all URLs, shortened links, and domain mentions from the input message.
- The system MUST generate a risk score between 0 and 100 for each submitted message.
- The system MUST assign a primary scam category (e.g., Phishing, Financial Manipulation, Legitimate) to each message.
- The system MUST provide a short, human-readable explanation for the assigned risk score and category.
- The system MUST highlight suspicious phrases or elements within the input text to support the explanation.
- The system MUST perform comprehensive URL analysis, including:
    - Domain structure analysis (typosquatting, lookalike, excessive subdomains, unusual TLDs).
    - HTTPS and certificate signal checks.
    - Domain age and reputation analysis (simulated).
    - Safe redirection behavior analysis (HEAD requests only).
    - Content semantic fingerprinting (minimal metadata, login forms, brand impersonation) without executing JavaScript or submitting forms.
- The system MUST display URL analysis results, including URL risk score, verdict, signals, and reasoning.

## 6. Non-Functional Requirements
- **Performance**: The system should provide analysis results with reasonable latency suitable for interactive use (hackathon context).
- **Scalability**: The architecture should conceptually support scaling of backend services and AI models (though not fully implemented for a hackathon).
- **Security**:
    - The system MUST NOT blindly open or execute URLs.
    - The system MUST NOT download files from analyzed URLs.
    - The system MUST NOT trust DNS or appearance alone for URL verification.
    - URL verification must be SAFE, READ-ONLY, and ANALYTICAL.
    - The system must adhere to rate limiting on API endpoints.
- **Usability**: The web interface should be intuitive and easy to use for submitting messages and interpreting results.
- **Maintainability**: The codebase should be modular and well-structured.
- **Explainability**: The system should provide clear reasoning for its predictions, especially for high-risk detections.

## 7. Constraints & Assumptions
- **Pre-trained Models**: Assumes the availability of pre-trained FastText, Sentence Transformer, DistilBERT, and FAISS index models (or simplified placeholders for the hackathon).
- **External APIs**: No reliance on external commercial APIs for threat intelligence beyond what's simulated or locally available for the hackathon.
- **Deployment Environment**: Assumes a Docker-friendly environment for deployment.
- **Input Language**: Primarily focused on English and Hinglish text.
- **Resource Limits**: Designed for typical hackathon resource availability (e.g., CPU-based inference for simplified models, rather than GPU).

## 8. Future Enhancements
- Integration with real-time threat intelligence feeds for domain reputation and blacklisting.
- User feedback mechanism for model improvement.
- Support for additional languages.
- Advanced visualization of model confidence and attention.
- Integration with external security tools (e.g., sandboxing for dynamic URL analysis).
- Implementation of user accounts and history.
- Real-time monitoring and alerting for detected scams.
- Container orchestration for production deployment.
- Continuous integration/continuous deployment (CI/CD) pipeline.
- Improved handling of QR code scanning (e.g., via image processing or external services).
- Fine-tuning of AI models with larger, more diverse datasets.
- More robust front-end validation and error handling.
- Accessibility improvements for the web application.