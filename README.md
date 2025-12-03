# Passport Name Extractor API
**Extract full names from passport images using Ollama Vision Language Models (VLM)**  
Deployed on Windows IIS + FastCGI • Powered by FastAPI, LangChain & Ollama

Perfect for KYC, onboarding automation, identity verification, and document-processing pipelines.

## Features
- Extract full name in English from any passport photo (MRZ or visual zone)
- Single & batch processing
- Two extraction methods:
  - `langchain` – recommended (better prompt control via LangChain + ChatOllama)
  - `direct` – raw Ollama `/api/generate` call (slightly faster)
- Accepts base64 images (with or without `data:image/...;base64,` prefix)
- CORS enabled for all origins
- Detailed logging (`logs/app.log`)
- Health-check endpoint with Ollama connectivity status
- Production-ready on Windows IIS using wfastcgi

## Endpoints

| Method | Endpoint            | Description                              |
|--------|---------------------|------------------------------------------|
| GET    | `/`                 | Service info & available endpoints       |
| GET    | `/health`           | Service + Ollama status                  |
| POST   | `/extract-name`     | Extract name from a single image         |
| POST   | `/batch-extract`    | Extract names from multiple images       |

OpenAPI docs (when running): `http://your-server/docs`

## Request Examples

### 1. Single Image
```bash
curl -X POST http://your-server/extract-name \
  -H "Content-Type: application/json" \
  -d '{
    "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "method": "langchain"
  }'
```
### Success response
```json 
{
  "success": true,
  "extracted_name": "Ahmed Mohamed Al-Sayed",
  "method_used": "langchain"
}
```
### 2. Batch Extraction (up to 100MB total payload)
```json
{
  "images": [
    { "id": "img_001", "data": "data:image/jpeg;base64,..." },
    { "id": "img_002", "data": "data:image/png;base64,..." }
  ],
  "method": "langchain"
}
```

