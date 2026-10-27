import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "knowledge_base.json",
)

_STOPWORDS = frozenset(
    {
        # English
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "must", "am", "of", "in",
        "to", "for", "with", "on", "at", "from", "by", "about", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "out", "off", "up", "down", "and", "but", "or", "nor", "not", "so",
        "yet", "both", "either", "neither", "each", "every", "all", "any",
        "few", "more", "most", "other", "some", "such", "no", "only", "own",
        "same", "than", "too", "very", "this", "that", "these", "those",
        "i", "me", "my", "myself", "we", "our", "ours", "you", "your",
        "he", "him", "his", "she", "her", "it", "its", "they", "them",
        "their", "what", "which", "who", "whom", "when", "where", "why",
        "how", "if", "then", "because", "while", "although", "though",
        "just", "also", "already", "still",
        # Hindi transliterated common words
        "ka", "ke", "ki", "ko", "se", "me", "hai", "hain", "tha", "the",
        "thi", "kya", "kaise", "kab", "kyun", "aur", "ya", "par", "nahi",
        "ho", "hota", "hoti", "karna", "kare", "karein", "mein", "yeh",
        "woh", "ek", "koi", "kuch",
    }
)


_FIELD_WEIGHTS = {
    "name": 5.0,
    "name_hi": 5.0,
    "scientific_name": 3.0,
    "category": 2.0,
    "topic": 5.0,
    "description": 2.0,
    "symptoms": 4.0,
    "damage_symptoms": 4.0,
    "causal_organism": 3.0,
    "management": 3.0,
    "recommended_practice": 2.0,
    "key_principles": 2.0,
    "cultivation_practices": 2.0,
    "fertilizer_recommendations": 2.0,
    "irrigation": 2.0,
    "eligibility": 2.0,
}



class KnowledgeBase:
    """Loads and searches the Annadata AI agricultural knowledge base."""

    def __init__(self, kb_path: Optional[str] = None) -> None:
        self.kb_path = kb_path or _DEFAULT_KB_PATH
        self._raw: Dict[str, Any] = {}
        self._entries: List[Dict[str, Any]] = []
        self._loaded = False


    def load(self) -> None:
        """Load and flatten the knowledge base JSON into searchable entries."""
        with open(self.kb_path, "r", encoding="utf-8") as fh:
            self._raw = json.load(fh)

        self._entries = []

        for crop in self._raw.get("crops", []):
            self._entries.append(
                self._build_crop_entry(crop)
            )
            for disease in crop.get("diseases", []):
                self._entries.append(
                    self._build_disease_entry(disease, crop)
                )
            for pest in crop.get("pests", []):
                self._entries.append(
                    self._build_pest_entry(pest, crop)
                )

        for item in self._raw.get("soil_management", []):
            self._entries.append(self._build_general_entry(item, "soil_management"))

        for item in self._raw.get("government_schemes", []):
            self._entries.append(self._build_scheme_entry(item))

        for item in self._raw.get("general_practices", []):
            self._entries.append(self._build_general_entry(item, "general_practice"))

        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


    @staticmethod
    def _build_crop_entry(crop: dict) -> dict:
        """Build a flat search entry for a crop."""
        return {
            "type": "crop",
            "id": crop.get("id", ""),
            "crop_name": crop.get("name", ""),
            "crop_name_hi": crop.get("name_hi", ""),
            "searchable_text": _flatten_to_text(crop, exclude_keys={"diseases", "pests"}),
            "data": crop,
            "sources": crop.get("sources", []),
        }

    @staticmethod
    def _build_disease_entry(disease: dict, parent_crop: dict) -> dict:
        return {
            "type": "disease",
            "id": disease.get("id", ""),
            "crop_name": parent_crop.get("name", ""),
            "crop_name_hi": parent_crop.get("name_hi", ""),
            "disease_name": disease.get("name", ""),
            "searchable_text": _flatten_to_text(disease),
            "data": disease,
            "sources": disease.get("sources", []),
        }

    @staticmethod
    def _build_pest_entry(pest: dict, parent_crop: dict) -> dict:
        return {
            "type": "pest",
            "id": pest.get("id", ""),
            "crop_name": parent_crop.get("name", ""),
            "crop_name_hi": parent_crop.get("name_hi", ""),
            "pest_name": pest.get("name", ""),
            "searchable_text": _flatten_to_text(pest),
            "data": pest,
            "sources": pest.get("sources", []),
        }

    @staticmethod
    def _build_scheme_entry(scheme: dict) -> dict:
        return {
            "type": "government_scheme",
            "id": scheme.get("id", ""),
            "searchable_text": _flatten_to_text(scheme),
            "data": scheme,
            "sources": scheme.get("sources", []),
        }

    @staticmethod
    def _build_general_entry(item: dict, entry_type: str) -> dict:
        return {
            "type": entry_type,
            "id": item.get("id", ""),
            "searchable_text": _flatten_to_text(item),
            "data": item,
            "sources": item.get("sources", []),
        }


    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.01,
        entry_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self._ensure_loaded()

        tokens = _tokenize(query)
        if not tokens:
            return []

        scored: List[Tuple[float, List[str], Dict[str, Any]]] = []

        for entry in self._entries:
            if entry_type and entry["type"] != entry_type:
                continue
            score, matched = self._score_entry(tokens, entry)
            if score >= min_score:
                scored.append((score, matched, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, matched, entry in scored[:top_k]:
            results.append(
                {
                    "entry": entry,
                    "score": round(score, 4),
                    "matched_keywords": matched,
                }
            )
        return results

    def _score_entry(
        self, query_tokens: List[str], entry: Dict[str, Any]
    ) -> Tuple[float, List[str]]:

        total_score = 0.0
        matched_keywords: set = set()
        data = entry.get("data", {})

        for field, weight in _FIELD_WEIGHTS.items():
            field_text = _extract_field_text(data, field)
            if not field_text:
                continue
            field_lower = field_text.lower()
            for token in query_tokens:
                if token in field_lower:
                    total_score += weight
                    matched_keywords.add(token)

        full_text = entry.get("searchable_text", "").lower()
        for token in query_tokens:
            if token in full_text:
                total_score += 1.0
                matched_keywords.add(token)

        if query_tokens:
            total_score = total_score / len(query_tokens)

        return total_score, sorted(matched_keywords)


    def get_crop_names(self) -> List[str]:
        self._ensure_loaded()
        return [
            e["crop_name"]
            for e in self._entries
            if e["type"] == "crop" and e.get("crop_name")
        ]

    def get_all_entries(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return list(self._entries)

    def get_entry_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        for entry in self._entries:
            if entry.get("id") == entry_id:
                return entry
        return None

    def get_metadata(self) -> Dict[str, Any]:
        self._ensure_loaded()
        return dict(self._raw.get("metadata", {}))

    def get_schemes(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        return [e for e in self._entries if e["type"] == "government_scheme"]

    def get_diseases_for_crop(self, crop_name: str) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        crop_lower = crop_name.strip().lower()
        return [
            e
            for e in self._entries
            if e["type"] == "disease"
            and e.get("crop_name", "").lower() == crop_lower
        ]

    def get_pests_for_crop(self, crop_name: str) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        crop_lower = crop_name.strip().lower()
        return [
            e
            for e in self._entries
            if e["type"] == "pest"
            and e.get("crop_name", "").lower() == crop_lower
        ]



def _tokenize(text: str) -> List[str]:
    text = text.lower()
    # Keep alphanumeric and some Hindi Unicode characters
    tokens = re.findall(r"[a-z0-9ऀ-ॿ]+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _flatten_to_text(obj: Any, exclude_keys: Optional[set] = None) -> str:
    parts: list = []
    _flatten_recursive(obj, parts, exclude_keys or set())
    return " ".join(parts)


def _flatten_recursive(obj: Any, parts: list, exclude_keys: set) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in exclude_keys:
                continue
            _flatten_recursive(value, parts, exclude_keys)
    elif isinstance(obj, list):
        for item in obj:
            _flatten_recursive(item, parts, exclude_keys)
    elif isinstance(obj, str):
        parts.append(obj)
    elif isinstance(obj, (int, float)):
        parts.append(str(obj))


def _extract_field_text(data: dict, field: str) -> str:
    value = data.get(field)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    if isinstance(value, dict):
        return _flatten_to_text(value)
    return str(value)


if __name__ == "__main__":
    kb = KnowledgeBase()
    kb.load()
    print(f"Loaded {len(kb.get_all_entries())} entries from knowledge base.")
    print(f"Crops: {kb.get_crop_names()}")
    print()

    sample_queries = [
        "How to manage blast disease in rice?",
        "What fertilizer should I use for wheat?",
        "Tell me about PM Kisan scheme",
        "Fall armyworm in maize",
        "tomato leaf curl virus management",
    ]
    for q in sample_queries:
        results = kb.search(q, top_k=3)
        print(f"Q: {q}")
        for r in results:
            e = r["entry"]
            print(f"  → [{e['type']}] {e.get('crop_name', '')} "
                  f"{e.get('disease_name', '')}{e.get('pest_name', '')} "
                  f"| score={r['score']} | keywords={r['matched_keywords']}")
        print()
