import os
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from modules.knowledge_base import KnowledgeBase



_MIN_SEARCH_SCORE = 0.5      # Minimum score to consider a result relevant
_TOP_K = 5                    # Max entries to retrieve per query
_UNCERTAINTY_THRESHOLD = 0.5  # If best score is below this, flag uncertainty

# Supported Indian languages with their codes and names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "bn": "Bengali",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "raj": "Rajasthani",
    "go": "Konkani",
    "mni": "Manipuri",
    "sd": "Sindhi",
    "sa": "Sanskrit",
    "ks": "Kashmiri",
    "ne": "Nepali",
    "doi": "Dogri",
    "bho": "Bhojpuri",
}

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are Annadata AI, an agricultural assistant for Indian farmers.

    RULES:
    1. Answer ONLY using the information provided in the CONTEXT section below.
    2. If the context does not contain enough information, say so clearly and
       recommend the farmer consult their nearest Krishi Vigyan Kendra (KVK)
       or agricultural extension officer.
    3. Keep answers practical, simple, and easy to understand.
    4. Always mention the sources at the end of your answer.
    5. Do NOT present uncertain diagnoses as confirmed facts.
    6. Use bullet points and short paragraphs for readability.
    7. If the farmer's question is ambiguous, ask a clarifying question.
    8. Include Hindi names where available to help the farmer identify crops,
       diseases, or pests.
    9. CRITICAL: Respond in the SAME LANGUAGE as the farmer's question.
""")

class LLMBackend(ABC):
    """Abstract base class for LLM backends (Evon, Gemini, OpenAI-compatible)."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        original_query: str,
        language: str,
    ) -> str:
        """Generate an answer from the LLM."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name for logging."""
        pass


class EvonBackend(LLMBackend):
    """
    Gnani Evon v3.3 LLM backend via OpenAI-compatible API.

    Works with vLLM or SGLang hosted endpoints.
    """

    def __init__(
        self,
        api_key: str,
        endpoint_url: str = "https://api.gnani.ai/v1/chat/completions",
        model_id: str = "evon-v3.3",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "evon"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        original_query: str,
        language: str,
    ) -> str:
        """Generate answer using Evon LLM via OpenAI-compatible API."""
        import requests

        lang_name = SUPPORTED_LANGUAGES.get(language, language)
        full_user_prompt = textwrap.dedent(f"""\
            ## CONTEXT (Retrieved Agricultural Information)

            {context}

            ## FARMER'S QUESTION (in {lang_name})

            {original_query}

            ## INSTRUCTIONS

            Answer the farmer's question based ONLY on the context above.
            Respond in {lang_name} (the same language as the question).
            End with a 'Sources:' section listing the references.
        """)

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_prompt},
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as exc:
            raise RuntimeError(f"Evon API error: {exc}") from exc


class GeminiBackend(LLMBackend):
    """Google Gemini LLM backend."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "gemini"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        original_query: str,
        language: str,
    ) -> str:
        """Generate answer using Google Gemini API."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            lang_name = SUPPORTED_LANGUAGES.get(language, language)
            full_prompt = textwrap.dedent(f"""\
                {system_prompt}

                ## CONTEXT (Retrieved Agricultural Information)

                {context}

                ## FARMER'S QUESTION (in {lang_name})

                {original_query}

                ## INSTRUCTIONS

                Answer the farmer's question based ONLY on the context above.
                Respond in {lang_name} (the same language as the question).
                End with a 'Sources:' section listing the references.
            """)

            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(full_prompt)
            return response.text

        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Gemini API error: {exc}") from exc


class OpenAICompatibleBackend(LLMBackend):
    """Generic OpenAI-compatible backend (OpenAI, Azure, local vLLM, etc.)."""

    def __init__(
        self,
        api_key: str,
        endpoint_url: str,
        model_id: str,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.model_id = model_id
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "openai-compatible"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        original_query: str,
        language: str,
    ) -> str:
        """Generate answer using OpenAI-compatible API."""
        import requests

        lang_name = SUPPORTED_LANGUAGES.get(language, language)
        full_user_prompt = textwrap.dedent(f"""\
            ## CONTEXT (Retrieved Agricultural Information)

            {context}

            ## FARMER'S QUESTION (in {lang_name})

            {original_query}

            ## INSTRUCTIONS

            Answer the farmer's question based ONLY on the context above.
            Respond in {lang_name} (the same language as the question).
            End with a 'Sources:' section listing the references.
        """)

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_prompt},
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as exc:
            raise RuntimeError(f"OpenAI-compatible API error: {exc}") from exc


# ─── Language Detection & Translation ────────────────────────────────────────

class LanguageHandler:
    """
    Handles language detection and translation for multilingual support.

    Currently uses a simple heuristic for detection and can be extended
    with proper libraries (langdetect, fasttext) or external APIs.
    """

    def __init__(self, translator_api_key: Optional[str] = None) -> None:
        self.translator_api_key = translator_api_key

    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.

        Returns a language code (e.g., 'te', 'hi', 'en').
        Falls back to 'en' if detection fails.
        """
        # Simple heuristic: check for Unicode ranges of Indian scripts
        if not text:
            return "en"

        # Count characters in different Unicode ranges
        telugu_range = sum(1 for c in text if 'ఀ' <= c <= '౿')
        tamil_range = sum(1 for c in text if '஀' <= c <= '௿')
        kannada_range = sum(1 for c in text if 'ಀ' <= c <= '೿')
        malayalam_range = sum(1 for c in text if 'ഀ' <= c <= 'ൿ')
        hindi_range = sum(1 for c in text if 'ऀ' <= c <= 'ॿ')
        bengali_range = sum(1 for c in text if 'ঀ' <= c <= '৿')
        gujarati_range = sum(1 for c in text if '઀' <= c <= '૿')
        punjabi_range = sum(1 for c in text if '਀' <= c <= '੿')

        # Find the dominant script
        ranges = {
            "te": telugu_range,
            "ta": tamil_range,
            "kn": kannada_range,
            "ml": malayalam_range,
            "hi": hindi_range,
            "bn": bengali_range,
            "gu": gujarati_range,
            "pa": punjabi_range,
        }

        max_lang = max(ranges, key=ranges.get)
        if ranges[max_lang] > 0:
            return max_lang

        # If no Indian script detected, assume English
        return "en"

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text to English for KB search.

        If source_lang is 'en', returns text as-is.
        Otherwise, uses translation service or LLM fallback.
        """
        if source_lang == "en":
            return text

        # Try Google Translate API if key is available
        if self.translator_api_key:
            return self._translate_with_google(text, source_lang, "en")

        # Fallback: ask the LLM to translate during the generation step
        # This is handled in the LLM prompt, not here
        # For now, return the original text and let LLM handle it
        return text

    def _translate_with_google(
        self, text: str, source_lang: str, target_lang: str
    ) -> str:
        """Translate using Google Cloud Translation API."""
        import requests

        url = "https://translation.googleapis.com/language/translate/v2"
        params = {
            "key": self.translator_api_key,
            "q": text,
            "source": source_lang,
            "target": target_lang,
        }

        try:
            response = requests.post(url, params=params, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result["data"]["translations"][0]["translatedText"]
        except Exception as exc:
            print(f"[LanguageHandler] Translation error: {exc}")
            return text


# ─── Main Answer Engine ───────────────────────────────────────────────────────

class AnswerEngine:
    """
    Multilingual answer engine for agricultural questions.

    Parameters
    ----------
    kb : KnowledgeBase, optional
        A pre-loaded KnowledgeBase instance.
    llm_backend : LLMBackend, optional
        The LLM backend to use (Evon, Gemini, or OpenAI-compatible).
        If not set, falls back to template-based generation.
    language_handler : LanguageHandler, optional
        Handler for language detection and translation.
    """

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        llm_backend: Optional[LLMBackend] = None,
        language_handler: Optional[LanguageHandler] = None,
    ) -> None:
        self.kb = kb or KnowledgeBase()
        self.kb.load()
        self.llm_backend = llm_backend
        self.language_handler = language_handler or LanguageHandler()

    @classmethod
    def with_evon(
        cls,
        api_key: str,
        endpoint_url: str = "https://api.gnani.ai/v1/chat/completions",
        model_id: str = "evon-v3.3",
        kb: Optional[KnowledgeBase] = None,
        translator_api_key: Optional[str] = None,
    ) -> "AnswerEngine":
        """Create an AnswerEngine configured with Evon LLM backend."""
        backend = EvonBackend(
            api_key=api_key,
            endpoint_url=endpoint_url,
            model_id=model_id,
        )
        lang_handler = LanguageHandler(translator_api_key=translator_api_key)
        return cls(kb=kb, llm_backend=backend, language_handler=lang_handler)

    @classmethod
    def with_gemini(
        cls,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        kb: Optional[KnowledgeBase] = None,
        translator_api_key: Optional[str] = None,
    ) -> "AnswerEngine":
        """Create an AnswerEngine configured with Google Gemini backend."""
        backend = GeminiBackend(api_key=api_key, model_name=model_name)
        lang_handler = LanguageHandler(translator_api_key=translator_api_key)
        return cls(kb=kb, llm_backend=backend, language_handler=lang_handler)

    # ── Public API ─────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a farmer's question with multilingual support.

        Parameters
        ----------
        question : str
            The farmer's question (can be in any supported language).
        language : str, optional
            ISO 639-1 language code (e.g., 'te', 'hi'). If not provided,
            the engine will attempt to detect the language.

        Returns
        -------
        dict with keys:
            question        — the original question
            answer          — the generated answer text (in original language)
            sources         — list of source references
            retrieved       — list of retrieved KB entries
            confidence      — 'high', 'medium', or 'low'
            method          — 'template', 'evon', 'gemini', etc.
            language        — detected or provided language code
            language_name   — full language name
        """
        if not question or not question.strip():
            return self._empty_response(question)

        # Step 1 — Detect or use provided language
        detected_lang = language or self.language_handler.detect_language(question)
        lang_name = SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)

        # Step 2 — Translate to English for KB search (if needed)
        english_query = self.language_handler.translate_to_english(
            question, detected_lang
        )

        # Step 3 — Retrieve relevant information from English KB
        results = self.kb.search(english_query, top_k=_TOP_K, min_score=0.01)
        relevant = [r for r in results if r["score"] >= _MIN_SEARCH_SCORE]

        # Step 4 — Determine confidence
        confidence = self._assess_confidence(results, relevant)

        # Step 5 — Generate answer
        if self.llm_backend and confidence != "low" and relevant:
            answer_text, method = self._generate_llm(
                question, relevant, confidence, detected_lang, lang_name
            )
        else:
            answer_text, method = self._generate_template(
                question, relevant, confidence, detected_lang
            )

        # Step 6 — Collect sources
        sources = self._collect_sources(relevant)

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "retrieved": [
                {
                    "type": r["entry"]["type"],
                    "id": r["entry"]["id"],
                    "score": r["score"],
                    "matched_keywords": r["matched_keywords"],
                }
                for r in relevant
            ],
            "confidence": confidence,
            "method": method,
            "language": detected_lang,
            "language_name": lang_name,
        }

    # ── Confidence assessment ──────────────────────────────────────────────

    @staticmethod
    def _assess_confidence(
        all_results: List[dict], relevant: List[dict]
    ) -> str:
        if not all_results:
            return "low"
        best = all_results[0]["score"]
        if best >= 3.0 and len(relevant) >= 2:
            return "high"
        if best >= _UNCERTAINTY_THRESHOLD:
            return "medium"
        return "low"

    # ── Template-based generation ──────────────────────────────────────────

    def _generate_template(
        self,
        question: str,
        relevant: List[dict],
        confidence: str,
        language: str,
    ) -> tuple:
        """Build a structured answer from retrieved entries without an LLM."""
        if confidence == "low" or not relevant:
            return self._low_confidence_answer(question, language), "template"

        parts: List[str] = []

        for r in relevant:
            entry = r["entry"]
            entry_type = entry["type"]
            data = entry["data"]

            if entry_type == "crop":
                parts.append(self._format_crop(data, question))
            elif entry_type == "disease":
                parts.append(self._format_disease(data, entry.get("crop_name", "")))
            elif entry_type == "pest":
                parts.append(self._format_pest(data, entry.get("crop_name", "")))
            elif entry_type == "government_scheme":
                parts.append(self._format_scheme(data))
            elif entry_type in ("soil_management", "general_practice"):
                parts.append(self._format_general(data))

        answer_text = "\n\n---\n\n".join(parts)

        if confidence == "medium":
            # Add uncertainty note for medium-confidence answers
            answer_text += (
                "\n\n⚠️ **Note:** The information above is the closest match found "
                "in our knowledge base. If this does not fully answer your question, "
                "please provide more details or consult your nearest Krishi Vigyan "
                "Kendra (KVK) or agricultural extension officer."
            )

        return answer_text, "template"

    # ── Formatters for each entry type ─────────────────────────────────────

    @staticmethod
    def _format_crop(data: dict, question: str) -> str:
        """Format a crop entry, selecting sections relevant to the question."""
        q_lower = question.lower()
        name = data.get("name", "Unknown Crop")
        name_hi = data.get("name_hi", "")
        header = f"## 🌾 {name}"
        if name_hi:
            header += f" ({name_hi})"

        sections = [header]

        # Always include basic info
        sections.append(
            f"- **Scientific name:** {data.get('scientific_name', 'N/A')}\n"
            f"- **Category:** {data.get('category', 'N/A')}\n"
            f"- **Season:** {', '.join(data.get('season', []))}\n"
            f"- **Sowing:** {', '.join(data.get('sowing_months', []))}\n"
            f"- **Harvesting:** {', '.join(data.get('harvesting_months', []))}\n"
            f"- **Major states:** {', '.join(data.get('major_states', []))}\n"
            f"- **Soil type:** {data.get('soil_type', 'N/A')}\n"
            f"- **Soil pH:** {data.get('soil_ph', 'N/A')}\n"
            f"- **Yield potential:** {data.get('yield_potential_tonnes_per_hectare', 'N/A')} tonnes/ha"
        )

        # Cultivation practices
        cultivation_keywords = [
            "cultivation", "sowing", "planting", "seed", "nursery",
            "transplant", "how to grow", "land preparation", "spacing",
        ]
        if any(kw in q_lower for kw in cultivation_keywords) or "cultivat" in q_lower:
            cp = data.get("cultivation_practices", {})
            if cp:
                sections.append("### Cultivation Practices")
                for key, val in cp.items():
                    label = key.replace("_", " ").title()
                    sections.append(f"- **{label}:** {val}")

        # Fertilizer
        fert_keywords = ["fertiliz", "nutrient", "npk", "nitrogen", "phospho",
                         "potass", "urea", "dap", "manure", "compost"]
        if any(kw in q_lower for kw in fert_keywords):
            fr = data.get("fertilizer_recommendations", {})
            if fr:
                sections.append("### Fertilizer Recommendations")
                sections.append(
                    f"- **Nitrogen:** {fr.get('nitrogen_kg_per_hectare', '-')} kg/ha\n"
                    f"- **Phosphorus:** {fr.get('phosphorus_kg_per_hectare', '-')} kg/ha\n"
                    f"- **Potassium:** {fr.get('potassium_kg_per_hectare', '-')} kg/ha\n"
                    f"- **Schedule:** {fr.get('application_schedule', '-')}\n"
                    f"- **Micronutrients:** {fr.get('micronutrients', '-')}\n"
                    f"- **Organic manure:** {fr.get('organic_manure', '-')}"
                )

        # Irrigation
        irr_keywords = ["irrigat", "water", "drip", "sprinkler", "flood",
                        "moisture", "paani", "sinchai"]
        if any(kw in q_lower for kw in irr_keywords):
            irr = data.get("irrigation", {})
            if irr:
                sections.append("### Irrigation")
                sections.append(
                    f"- **Method:** {irr.get('method', '-')}\n"
                    f"- **Critical stages:** {', '.join(irr.get('critical_stages', []))}\n"
                    f"- **Schedule:** {irr.get('schedule', '-')}"
                )
                if irr.get("note"):
                    sections.append(f"- **Note:** {irr['note']}")

        # If no specific section was triggered, include all major sections
        if len(sections) <= 2:
            cp = data.get("cultivation_practices", {})
            if cp:
                sections.append("### Cultivation Practices")
                for key, val in cp.items():
                    label = key.replace("_", " ").title()
                    sections.append(f"- **{label}:** {val}")

            fr = data.get("fertilizer_recommendations", {})
            if fr:
                sections.append("### Fertilizer Recommendations")
                sections.append(
                    f"- **N:P:K** = {fr.get('nitrogen_kg_per_hectare', '-')}:"
                    f"{fr.get('phosphorus_kg_per_hectare', '-')}:"
                    f"{fr.get('potassium_kg_per_hectare', '-')} kg/ha\n"
                    f"- **Schedule:** {fr.get('application_schedule', '-')}"
                )

            irr = data.get("irrigation", {})
            if irr:
                sections.append("### Irrigation")
                sections.append(
                    f"- **Method:** {irr.get('method', '-')}\n"
                    f"- **Schedule:** {irr.get('schedule', '-')}"
                )

        return "\n".join(sections)

    @staticmethod
    def _format_disease(data: dict, crop_name: str) -> str:
        name = data.get("name", "Unknown Disease")
        name_hi = data.get("name_hi", "")
        header = f"## 🦠 {name}"
        if name_hi:
            header += f" ({name_hi})"
        if crop_name:
            header += f" — in {crop_name}"

        mgmt = data.get("management", {})

        parts = [
            header,
            f"- **Caused by:** {data.get('causal_organism', 'N/A')}",
            f"- **Symptoms:** {data.get('symptoms', 'N/A')}",
            f"- **Favorable conditions:** {data.get('favorable_conditions', 'N/A')}",
            "",
            "### Management",
            f"- **Cultural control:** {mgmt.get('cultural', 'N/A')}",
            f"- **Chemical control:** {mgmt.get('chemical', 'N/A')}",
            f"- **Biological control:** {mgmt.get('biological', 'N/A')}",
        ]
        return "\n".join(parts)

    @staticmethod
    def _format_pest(data: dict, crop_name: str) -> str:
        name = data.get("name", "Unknown Pest")
        name_hi = data.get("name_hi", "")
        header = f"## 🐛 {name}"
        if name_hi:
            header += f" ({name_hi})"
        if crop_name:
            header += f" — in {crop_name}"

        mgmt = data.get("management", {})

        parts = [
            header,
            f"- **Scientific name:** {data.get('scientific_name', 'N/A')}",
            f"- **Damage symptoms:** {data.get('damage_symptoms', 'N/A')}",
            f"- **Favorable conditions:** {data.get('favorable_conditions', 'N/A')}",
            f"- **Economic threshold:** {data.get('economic_threshold', 'N/A')}",
            "",
            "### Management",
            f"- **Cultural control:** {mgmt.get('cultural', 'N/A')}",
            f"- **Chemical control:** {mgmt.get('chemical', 'N/A')}",
            f"- **Biological control:** {mgmt.get('biological', 'N/A')}",
        ]
        return "\n".join(parts)

    @staticmethod
    def _format_scheme(data: dict) -> str:
        name = data.get("name", "Unknown Scheme")
        name_hi = data.get("name_hi", "")
        header = f"## 🏛️ {name}"
        if name_hi:
            header += f" ({name_hi})"

        parts = [
            header,
            f"- **Description:** {data.get('description', 'N/A')}",
            f"- **Eligibility:** {data.get('eligibility', 'N/A')}",
            f"- **How to apply:** {data.get('how_to_apply', 'N/A')}",
        ]
        if data.get("website"):
            parts.append(f"- **Website:** {data['website']}")
        return "\n".join(parts)

    @staticmethod
    def _format_general(data: dict) -> str:
        topic = data.get("topic", "Information")
        name_hi = data.get("name_hi", "")
        header = f"## 📋 {topic}"
        if name_hi:
            header += f" ({name_hi})"

        parts = [header]

        if data.get("description"):
            parts.append(data["description"])

        if data.get("recommended_practice"):
            parts.append(f"\n**Recommended Practice:** {data['recommended_practice']}")

        if data.get("key_principles"):
            parts.append("\n**Key Principles:**")
            for p in data["key_principles"]:
                parts.append(f"- {p}")

        if data.get("parameters_tested"):
            parts.append(f"\n**Parameters tested:** {', '.join(data['parameters_tested'])}")

        if data.get("where_to_test"):
            parts.append(f"\n**Where to test:** {data['where_to_test']}")

        if data.get("benefits"):
            parts.append("\n**Benefits:**")
            for b in data["benefits"]:
                parts.append(f"- {b}")

        return "\n".join(parts)

    # ── Low-confidence / empty responses ───────────────────────────────────

    @staticmethod
    def _low_confidence_answer(question: str, language: str = "en") -> str:
        # In template mode, we can only provide English fallback
        return (
            "I'm sorry, I could not find enough reliable information in the "
            "agricultural knowledge base to answer your question confidently.\n\n"
            f"**Your question:** {question}\n\n"
            "**What you can do:**\n"
            "1. Try rephrasing your question with specific crop or disease names.\n"
            "2. Contact your nearest **Krishi Vigyan Kendra (KVK)** for expert guidance.\n"
            "3. Call the **Kisan Call Centre** at **1800-180-1551** (toll-free) "
            "for agricultural advice.\n"
            "4. Visit the **Farmer Portal** at https://farmer.gov.in for more resources.\n\n"
            "We want to give you accurate information, so we prefer to admit "
            "uncertainty rather than provide incorrect guidance."
        )

    @staticmethod
    def _empty_response(question: str) -> Dict[str, Any]:
        return {
            "question": question or "",
            "answer": "Please enter a valid agricultural question so I can help you.",
            "sources": [],
            "retrieved": [],
            "confidence": "low",
            "method": "template",
            "language": "en",
            "language_name": "English",
        }

    # ── Source collection ──────────────────────────────────────────────────

    @staticmethod
    def _collect_sources(relevant: List[dict]) -> List[str]:
        seen = set()
        sources = []
        for r in relevant:
            for src in r["entry"].get("sources", []):
                if src not in seen:
                    seen.add(src)
                    sources.append(src)
        return sources

    # ── LLM-based generation ───────────────────────────────────────────────

    def _generate_llm(
        self,
        question: str,
        relevant: List[dict],
        confidence: str,
        language: str,
        lang_name: str,
    ) -> tuple:
        """
        Generate a natural-language answer using the configured LLM backend.

        Falls back to template-based generation if the API call fails.
        """
        if not self.llm_backend:
            return self._generate_template(question, relevant, confidence, language)

        # Build context from retrieved entries
        context_parts = []
        for i, r in enumerate(relevant, 1):
            entry = r["entry"]
            data = entry["data"]
            context_parts.append(
                f"--- Entry {i} (type: {entry['type']}, score: {r['score']}) ---\n"
                f"{_dict_to_readable(data)}"
            )
        context_text = "\n\n".join(context_parts)

        try:
            answer_text = self.llm_backend.generate(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt="",  # Included in full prompt inside generate()
                context=context_text,
                original_query=question,
                language=language,
            )
            return answer_text, self.llm_backend.name

        except Exception as exc:
            print(f"[AnswerEngine] LLM error: {exc}. Falling back to template.")
            return self._generate_template(question, relevant, confidence, language)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _dict_to_readable(d: Any, indent: int = 0) -> str:
    """Convert a nested dict/list to a readable indented text block."""
    lines = []
    prefix = "  " * indent
    if isinstance(d, dict):
        for key, val in d.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dict_to_readable(val, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {val}")
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (dict, list)):
                lines.append(_dict_to_readable(item, indent))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{d}")
    return "\n".join(lines)


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with template backend (no API key needed)
    engine = AnswerEngine()

    test_questions = [
        "How to manage blast disease in rice?",
        "What fertilizer should I use for wheat?",
        "Tell me about PM Kisan scheme",
        "How to control fall armyworm in maize?",
        "నా వరి పంట ఆకులు పసుపు రంగులోకి మారుతున్నాయి. ఏం చేయాలి?",  # Telugu
        "मेरी गेहूं की फसल में पीला रतुआ रोग लग गया है। क्या करूं?",  # Hindi
    ]

    for q in test_questions:
        print("=" * 70)
        result = engine.answer(q)
        print(f"Q: {result['question']}")
        print(f"Language: {result['language_name']} ({result['language']})")
        print(f"Confidence: {result['confidence']} | Method: {result['method']}")
        print(f"Sources: {result['sources']}")
        print()
        print(result["answer"])
        print()
