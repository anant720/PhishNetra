# PhishNetra - API Reference

Complete API documentation for RiskAnalyzer AI's REST endpoints.

## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Currently, no authentication is required. For production deployments, consider adding API keys or OAuth.

## Rate Limiting
- 100 requests per minute per IP address
- Configurable via environment variables

---

## Analyze Text

Analyze text for scam detection using all AI models.

### Endpoint
```
POST /analyze
```

### Request Body
```json
{
  "text": "URGENT: Your account has been suspended. Click here to verify: http://fakebank.com/verify",
  "include_explainability": true,
  "format_type": "full"
}
```

### Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text to analyze (1-10000 characters) |
| `include_explainability` | boolean | No | Include detailed explanations (default: true) |
| `format_type` | string | No | Response format: "full", "basic", "minimal" (default: "full") |

### Response (Full Format)
```json
{
  "risk_score": 85.5,
  "confidence": 0.82,
  "threat_category": "phishing",
  "reasoning": "High-risk phishing attempt detected with suspicious URL and urgent language",
  "model_confidence_breakdown": {
    "fasttext": 0.75,
    "sentence_transformer": 0.85,
    "distilbert": 0.88,
    "similarity": 0.78
  },
  "explanation": {
    "narrative_explanation": "This message shows strong indicators of being a scam...",
    "highlighted_phrases": [
      {
        "phrase": "URGENT",
        "category": "urgency_pressure",
        "severity": "high",
        "explanation": "Creates false urgency to pressure quick action"
      }
    ],
    "risk_factors": [
      {
        "factor": "contains_urls",
        "description": "Contains 1 URL(s) - verify before clicking",
        "score": 0.7,
        "source": "text_analysis"
      }
    ],
    "recommendations": [
      "Do not respond to this message or provide any personal information",
      "Do not click any links or download attachments"
    ]
  },
  "metadata": {
    "text_length": 89,
    "processed_text_length": 89,
    "models_used": ["fasttext", "sentence_transformer", "distilbert", "similarity"],
    "request_id": "generated_request_id"
  }
}
```

### Response Codes
- `200`: Success
- `400`: Invalid request (missing text, too long, etc.)
- `429`: Rate limit exceeded
- `500`: Internal server error

---

## Batch Analysis

Analyze multiple texts in a single request.

### Endpoint
```
POST /analyze/batch
```

### Request Body
```json
{
  "texts": [
    "URGENT: Your account has been suspended...",
    "Congratulations! You've won $1,000,000...",
    "Hi mom, I need money urgently..."
  ],
  "include_explainability": false
}
```

### Parameters
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | array | Yes | Array of texts to analyze (max 50) |
| `include_explainability` | boolean | No | Include detailed explanations (default: true) |

### Response
```json
{
  "results": [
    {
      "index": 0,
      "risk_score": 85.5,
      "confidence": 0.82,
      "threat_category": "phishing",
      "reasoning": "High-risk phishing attempt detected..."
    },
    {
      "index": 1,
      "error": "Invalid text input"
    }
  ],
  "total_processed": 2,
  "total_errors": 1
}
```

---

## Health Check

Check service health and component status.

### Endpoint
```
GET /health
```

### Response
```json
{
  "status": "healthy",
  "service": "RiskAnalyzer AI",
  "version": "1.0.0",
  "message": "Service is running and ready to analyze texts"
}
```

### Detailed Health Check
```
GET /health/detailed
```

### Response
```json
{
  "status": "healthy",
  "service": "RiskAnalyzer AI",
  "version": "1.0.0",
  "components": {
    "models": {
      "fasttext": {
        "status": "healthy",
        "info": {
          "model_type": "FastText",
          "vocabulary_size": 50000,
          "vector_dimension": 300
        }
      }
    },
    "preprocessing": "healthy",
    "explainability": "healthy",
    "decision_fusion": "healthy"
  },
  "overall_health": true
}
```

---

## Model Information

Get information about loaded AI models.

### Endpoint
```
GET /models/info
```

### Response
```json
{
  "models": {
    "fasttext": {
      "model_type": "FastText",
      "vocabulary_size": 50000,
      "vector_dimension": 300,
      "has_model": true
    },
    "sentence_transformer": {
      "model_type": "SentenceTransformer",
      "model_name": "all-MiniLM-L6-v2",
      "embedding_dimension": 384,
      "device": "cpu",
      "has_model": true
    },
    "distilbert": {
      "model_type": "DistilBERT",
      "model_path": "./models/distilbert_scam",
      "max_seq_length": 512,
      "num_classes": 2,
      "has_model": true
    },
    "similarity": {
      "index_type": "FAISS",
      "total_vectors": 10000,
      "embedding_dimension": 384,
      "has_embedding_model": true,
      "scam_texts_count": 10000
    },
    "decision_fusion": {
      "fusion_method": "weighted_ensemble",
      "model_weights": {
        "fasttext": 0.25,
        "sentence_transformer": 0.30,
        "distilbert": 0.35,
        "similarity": 0.10
      },
      "supported_categories": [
        "legitimate",
        "financial_scam",
        "phishing",
        "impersonation",
        "urgency_scam",
        "authority_scam",
        "social_engineering",
        "tech_support_scam",
        "unknown_scam"
      ]
    }
  },
  "total_models": 5
}
```

---

## Error Responses

### Standard Error Format
```json
{
  "detail": "Error description",
  "error_code": "ERROR_TYPE",
  "request_id": "generated_request_id"
}
```

### Common Errors
- `INVALID_TEXT`: Text is empty or too long
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `MODEL_UNAVAILABLE`: AI model not loaded
- `INTERNAL_ERROR`: Unexpected server error

---

## SDK Examples

### JavaScript/Node.js
```javascript
const axios = require('axios');

async function analyzeText(text) {
  try {
    const response = await axios.post('http://localhost:8000/api/v1/analyze', {
      text: text,
      include_explainability: true
    });

    console.log('Risk Score:', response.data.risk_score);
    console.log('Category:', response.data.threat_category);
    return response.data;
  } catch (error) {
    console.error('Analysis failed:', error.response.data);
  }
}
```

### Python
```python
import requests

def analyze_text(text):
    response = requests.post('http://localhost:8000/api/v1/analyze', json={
        'text': text,
        'include_explainability': True
    })

    if response.status_code == 200:
        data = response.json()
        print(f"Risk Score: {data['risk_score']}")
        print(f"Category: {data['threat_category']}")
        return data
    else:
        print(f"Error: {response.json()}")
        return None
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT: Your account has been suspended. Click here: http://fakebank.com/verify",
    "include_explainability": true
  }'
```

---

## Webhook Integration

For high-volume integrations, consider webhook callbacks:

```json
{
  "webhook_url": "https://your-app.com/webhook/riskanalyzer",
  "callback_events": ["analysis_complete", "high_risk_detected"]
}
```

---

## Performance Guidelines

- **Batch requests**: Use `/analyze/batch` for multiple texts
- **Minimal responses**: Set `format_type: "minimal"` for high throughput
- **Caching**: Implement client-side caching for repeated analyses
- **Rate limits**: Respect the 100 requests/minute limit
- **Timeouts**: Set client timeouts to 30 seconds

---

## Versioning

API versioning follows semantic versioning:
- `v1`: Current stable version
- Breaking changes will introduce new versions (v2, v3, etc.)

---

## Support

For API issues or questions:
- Check the health endpoint: `GET /health`
- View API documentation: `GET /docs` (Swagger UI)
- Check model status: `GET /models/info`
- Review logs for detailed error information