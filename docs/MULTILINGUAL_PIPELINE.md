Annadata AI — Multilingual Answer Pipeline

Overview

Annadata AI supports **22 Indian languages** through a multilingual question-answering pipeline. The knowledge base remains in English (single source of truth), while the LLM handles language understanding and response generation.

Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANNADATA AI PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                           │
│  │   FARMER     │  Speaks in local language (e.g., Telugu)                 │
│  │   (Voice)    │                                                           │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │  PRISMA STT  │  Gnani's Speech-to-Text                                   │
│  │  (Gnani)     │  Input: Audio in 22 languages                             │
│  │              │  Output: Text in same language                            │
│  └──────┬───────┘                                                           │
│         │ "నా వరి పంట ఆకులు పసుపు రంగులోకి మారుతున్నాయి"                    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │  LANGUAGE    │  Detect language from Unicode script ranges               │
│  │  DETECTION   │  Output: "te" (Telugu)                                    │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │  TRANSLATE   │  Translate to English for KB search                       │
│  │  (Optional)  │  - If translation API available → translate               │
│  │              │  - If not → LLM will handle in next step                  │
│  └──────┬───────┘                                                           │
│         │ "My rice crop leaves are turning yellow"                          │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │   ENGLISH    │  Keyword-based retrieval                                  │
│  │ KNOWLEDGE    │  8 crops, 14 diseases, 12 pests, 5 schemes               │
│  │    BASE      │  Sources: ICAR, KVK, ICRISAT                              │
│  └──────┬───────┘                                                           │
│         │ Returns: Rice disease entries, fertilizer info, etc.              │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │    EVON      │  Gnani Evon v3.3 LLM                                      │
│  │    LLM       │  Input: English KB context + Original Telugu query        │
│  │              │  Instruction: "Answer in Telugu"                          │
│  └──────┬───────┘                                                           │
│         │ "మీ వరి పంటలో ఆకులు పసుపుగా మారడం అనేక కారణాల వల్ల జరగవచ్చు..."        │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │  TIMBRE TTS  │  Gnani's Text-to-Speech                                   │
│  │  (Gnani)     │  Input: Telugu text                                       │
│  │              │  Output: Telugu audio                                     │
│  └──────┬───────┘                                                           │
│         │                                                                    │
│         ▼                                                                    │
│  ┌──────────────┐                                                           │
│  │   FARMER     │  Hears answer in Telugu                                   │
│  │   (Voice)    │                                                           │
│  └──────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. English Knowledge Base (Single Source of Truth)

**Why?**
- Easier to maintain and update
- Single version to verify against ICAR/KVK sources
- Avoids translation inconsistencies across 22 languages
- Reduces storage and complexity

**How it works:**
- KB contains agricultural knowledge in English
- Queries are translated to English for search
- LLM generates answers in the original language

### 2. LLM Handles Multilingual Output

**Why?**
- Evon v3.3 natively supports Indian languages
- No need for separate translation of the answer
- LLM can adapt tone and terminology to the language

**How it works:**
```
System Prompt: "You are Annadata AI... Respond in the SAME LANGUAGE as the farmer's question."

Context: [English KB entries about rice diseases]

User Query: "నా వరి పంట ఆకులు పసుపు రంగులోకి మారుతున్నాయి"

LLM Output: [Telugu answer based on English context]
```

### 3. Language Detection via Unicode

**Why?**
- Fast, no external API needed
- Works offline
- Sufficient for Indian scripts with distinct Unicode ranges

**Supported Scripts:**
| Language | Code | Unicode Range |
|----------|------|---------------|
| Telugu | te | U+0C00–U+0C7F |
| Tamil | ta | U+0B80–U+0BFF |
| Kannada | kn | U+0C80–U+0CFF |
| Malayalam | ml | U+0D00–U+0D7F |
| Hindi | hi | U+0900–U+097F |
| Bengali | bn | U+0980–U+09FF |
| Gujarati | gu | U+0A80–U+0AFF |
| Punjabi | pa | U+0A00–U+0A7F |

---

## Backend Configuration

### Option 1: Evon LLM (Recommended)

```python
from modules.answer_engine import AnswerEngine

# Configure with Gnani Evon
engine = AnswerEngine.with_evon(
    api_key="your-gnani-api-key",
    endpoint_url="https://api.gnani.ai/v1/chat/completions",
    model_id="evon-v3.3",
)

# Answer a Telugu question
result = engine.answer("నా వరి పంట ఆకులు పసుపు రంగులోకి మారుతున్నాయి. ఏం చేయాలి?")

print(result["answer"])      # Telugu answer
print(result["language"])    # "te"
print(result["sources"])     # ["ICAR", "KVK", ...]
```

### Option 2: Google Gemini

```python
engine = AnswerEngine.with_gemini(
    api_key="your-google-api-key",
    model_name="gemini-2.0-flash",
)
```

### Option 3: Any OpenAI-Compatible Backend

```python
from modules.answer_engine import AnswerEngine, OpenAICompatibleBackend

backend = OpenAICompatibleBackend(
    api_key="your-api-key",
    endpoint_url="https://your-llm-endpoint.com/v1/chat/completions",
    model_id="your-model-id",
)

engine = AnswerEngine(llm_backend=backend)
```

### Option 4: Template-Based (Offline, No LLM)

```python
engine = AnswerEngine()  # No LLM backend configured

# Falls back to template-based answers (English only)
result = engine.answer("How to manage blast disease in rice?")
```

---

## API Reference

### `AnswerEngine.answer(question, language=None)`

**Parameters:**
- `question` (str): The farmer's question in any supported language
- `language` (str, optional): ISO 639-1 language code. If not provided, auto-detected.

**Returns:**
```python
{
    "question": "నా వరి పంట ఆకులు పసుపు రంగులోకి మారుతున్నాయి",
    "answer": "మీ వరి పంటలో ఆకులు పసుపుగా మారడం అనేక కారణాల వల్ల జరగవచ్చు...",
    "sources": ["ICAR", "KVK Rice Research Station"],
    "retrieved": [
        {"type": "disease", "id": "rice_blast", "score": 4.5, ...}
    ],
    "confidence": "high",  # "high", "medium", or "low"
    "method": "evon",       # "evon", "gemini", "template", etc.
    "language": "te",
    "language_name": "Telugu"
}
```

---

## Integration with Member 1's Streamlit UI

### Input from STT
```python
# Member 1 receives audio, calls Prisma STT
telugu_text = prisma_stt(audio_bytes)  # "నా వరి పంట..."

# Pass to Member 2's answer engine
result = engine.answer(telugu_text)

# Extract answer for TTS
answer_text = result["answer"]      # Telugu answer
language = result["language"]       # "te"
```

### Output to TTS
```python
# Member 1 sends answer to Timbre TTS
audio_response = timbre_tts(answer_text, language=language)

# Play to farmer
play_audio(audio_response)
```

---

## Confidence Levels

| Level | Condition | Behavior |
|-------|-----------|----------|
| **High** | Top score ≥ 3.0 AND ≥ 2 relevant results | Direct answer from LLM/template |
| **Medium** | Top score ≥ 0.5 | Answer + uncertainty note |
| **Low** | Top score < 0.5 OR no results | Fallback: Suggest KVK, Kisan Call Centre |

---

## Error Handling

1. **LLM API Failure** → Falls back to template-based answer
2. **Language Detection Failure** → Defaults to English
3. **KB Search No Results** → Returns low-confidence fallback message

---

## Environment Variables

```bash
# For Evon LLM
GNANI_API_KEY=your-gnani-api-key
EVON_ENDPOINT_URL=https://api.gnani.ai/v1/chat/completions
EVON_MODEL_ID=evon-v3.3

# For Google Gemini (alternative)
GOOGLE_API_KEY=your-google-api-key

# For Translation API (optional)
GOOGLE_TRANSLATE_API_KEY=your-google-translate-api-key
```

---

## Files Structure

```
annadata-ai/
├── data/
│   └── knowledge_base.json      # English KB (single source of truth)
├── modules/
│   ├── __init__.py
│   ├── knowledge_base.py        # KB loading and search
│   └── answer_engine.py         # Multilingual answer generation
├── tests/
│   └── test_annadata.py         # 65 tests, all passing
├── docs/
│   └── MULTILINGUAL_PIPELINE.md # This document
└── requirements.txt
```

---

## Next Steps

1. **Get Evon API details** from your friend:
   - Endpoint URL
   - Model ID
   - API key format (Bearer token, X-API-Key, etc.)

2. **Test multilingual flow**:
   - Send Telugu/Hindi queries
   - Verify Evon responds in correct language
   - Measure latency

3. **Coordinate with Member 1**:
   - Share `modules/` and `data/` directories
   - Define API contract for STT → Engine → TTS flow

---

## Contact

**Member 2 (Knowledge Base & Answer Engine):** Akshaj

**Last Updated:** 2026-10-26
