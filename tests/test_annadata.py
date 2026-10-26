"""
test_annadata.py — Test Suite for Annadata AI (Member 2 Deliverables)

Tests cover:
  1. Knowledge base loading and structure validation
  2. Knowledge retrieval (search accuracy)
  3. Answer engine (template-based generation)
  4. Edge cases: ambiguous queries, empty queries, unsupported topics
  5. Answer validation against trusted sources

Run with:
    python -I -m pytest tests/test_annadata.py -v

Author : Member 2 – Knowledge Base & Answer Engine Developer
Project: Annadata AI
"""

import json
import os
import sys

import pytest

# Add project root to path so modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.knowledge_base import KnowledgeBase, _tokenize
from modules.answer_engine import AnswerEngine


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def kb():
    """Load the knowledge base once for the entire test module."""
    kb = KnowledgeBase()
    kb.load()
    return kb


@pytest.fixture(scope="module")
def engine():
    """Create an answer engine (template mode, no LLM backend)."""
    return AnswerEngine()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. KNOWLEDGE BASE — STRUCTURE AND LOADING
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnowledgeBaseLoading:
    """Verify the knowledge base loads correctly and has valid structure."""

    def test_load_successful(self, kb):
        entries = kb.get_all_entries()
        assert len(entries) > 0, "Knowledge base should have at least one entry."

    def test_metadata_present(self, kb):
        meta = kb.get_metadata()
        assert "title" in meta
        assert "version" in meta
        assert "supported_topics" in meta

    def test_crops_present(self, kb):
        crops = kb.get_crop_names()
        assert len(crops) >= 5, f"Expected at least 5 crops, got {len(crops)}"
        assert "Rice" in crops
        assert "Wheat" in crops
        assert "Cotton" in crops
        assert "Tomato" in crops
        assert "Maize (Corn)" in crops

    def test_entry_types_present(self, kb):
        entries = kb.get_all_entries()
        types = {e["type"] for e in entries}
        assert "crop" in types
        assert "disease" in types
        assert "pest" in types
        assert "government_scheme" in types

    def test_schemes_present(self, kb):
        schemes = kb.get_schemes()
        assert len(schemes) >= 3, "Expected at least 3 government schemes."
        scheme_names = [s["data"]["name"] for s in schemes]
        assert any("PM-Kisan" in n for n in scheme_names)
        assert any("Fasal Bima" in n or "PMFBY" in n for n in scheme_names)

    def test_diseases_have_sources(self, kb):
        entries = kb.get_all_entries()
        diseases = [e for e in entries if e["type"] == "disease"]
        for d in diseases:
            assert len(d["sources"]) > 0, (
                f"Disease {d['id']} ({d.get('disease_name', '')}) has no sources."
            )

    def test_pests_have_sources(self, kb):
        entries = kb.get_all_entries()
        pests = [e for e in entries if e["type"] == "pest"]
        for p in pests:
            assert len(p["sources"]) > 0, (
                f"Pest {p['id']} ({p.get('pest_name', '')}) has no sources."
            )

    def test_crops_have_required_fields(self, kb):
        entries = kb.get_all_entries()
        crops = [e for e in entries if e["type"] == "crop"]
        for c in crops:
            data = c["data"]
            assert "name" in data, f"Crop {c['id']} missing 'name'"
            assert "scientific_name" in data, f"Crop {c['id']} missing 'scientific_name'"
            assert "season" in data, f"Crop {c['id']} missing 'season'"
            assert "cultivation_practices" in data, f"Crop {c['id']} missing 'cultivation_practices'"
            assert "fertilizer_recommendations" in data, f"Crop {c['id']} missing 'fertilizer_recommendations'"
            assert "irrigation" in data, f"Crop {c['id']} missing 'irrigation'"

    def test_diseases_for_rice(self, kb):
        rice_diseases = kb.get_diseases_for_crop("Rice")
        assert len(rice_diseases) >= 2, "Rice should have at least 2 diseases."
        names = [d["disease_name"] for d in rice_diseases]
        assert any("Blast" in n for n in names), "Rice Blast should be in the KB."

    def test_pests_for_rice(self, kb):
        rice_pests = kb.get_pests_for_crop("Rice")
        assert len(rice_pests) >= 1, "Rice should have at least 1 pest."

    def test_get_entry_by_id(self, kb):
        entry = kb.get_entry_by_id("crop_001")
        assert entry is not None
        assert entry["crop_name"] == "Rice"

    def test_get_entry_by_invalid_id_returns_none(self, kb):
        entry = kb.get_entry_by_id("nonexistent_999")
        assert entry is None

    def test_json_file_is_valid(self):
        kb_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "knowledge_base.json"
        )
        with open(kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "crops" in data
        assert isinstance(data["crops"], list)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenizer:
    """Test the query tokenization logic."""

    def test_basic_tokenization(self):
        tokens = _tokenize("How to manage blast in rice?")
        assert "manage" in tokens
        assert "blast" in tokens
        assert "rice" in tokens

    def test_stopwords_removed(self):
        tokens = _tokenize("How to manage the blast disease in rice?")
        assert "how" not in tokens
        assert "to" not in tokens
        assert "the" not in tokens
        assert "in" not in tokens

    def test_empty_input(self):
        assert _tokenize("") == []

    def test_only_stopwords(self):
        tokens = _tokenize("the a an is are was were")
        assert tokens == []

    def test_short_tokens_removed(self):
        tokens = _tokenize("a an I")
        assert tokens == []

    def test_hindi_transliterated_stopwords_removed(self):
        tokens = _tokenize("yeh kya hai rice blast")
        assert "yeh" not in tokens
        assert "kya" not in tokens
        assert "hai" not in tokens
        assert "rice" in tokens
        assert "blast" in tokens


# ═══════════════════════════════════════════════════════════════════════════════
# 3. KNOWLEDGE RETRIEVAL — SEARCH ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchAccuracy:
    """Test that search returns relevant results for farming questions."""

    def test_rice_blast_query(self, kb):
        results = kb.search("How to manage blast disease in rice?")
        assert len(results) > 0
        top = results[0]
        # Top result should be the rice blast disease entry
        assert top["entry"]["type"] == "disease"
        assert "Blast" in top["entry"].get("disease_name", "")

    def test_wheat_fertilizer_query(self, kb):
        results = kb.search("What fertilizer to use for wheat?")
        assert len(results) > 0
        # Top result should involve wheat
        top = results[0]
        assert "Wheat" in top["entry"].get("crop_name", "")

    def test_pm_kisan_query(self, kb):
        results = kb.search("Tell me about PM Kisan scheme")
        assert len(results) > 0
        top = results[0]
        assert top["entry"]["type"] == "government_scheme"
        assert "PM-Kisan" in top["entry"]["data"]["name"]

    def test_fall_armyworm_query(self, kb):
        results = kb.search("Fall armyworm in maize")
        assert len(results) > 0
        top = results[0]
        assert top["entry"]["type"] == "pest"
        assert "Fall Armyworm" in top["entry"].get("pest_name", "")

    def test_tomato_leaf_curl_query(self, kb):
        results = kb.search("tomato leaf curl virus management")
        assert len(results) > 0
        # Should return the ToLCV disease entry
        found_tolcv = any(
            "Leaf Curl" in r["entry"].get("disease_name", "")
            for r in results
        )
        assert found_tolcv, "Should find Tomato Leaf Curl Virus entry"

    def test_irrigation_query(self, kb):
        results = kb.search("How to irrigate sugarcane?")
        assert len(results) > 0
        top = results[0]
        assert "Sugarcane" in top["entry"].get("crop_name", "")

    def test_soil_testing_query(self, kb):
        results = kb.search("How to test soil health?")
        assert len(results) > 0
        found_soil = any(
            r["entry"]["type"] == "soil_management"
            for r in results
        )
        assert found_soil, "Should return a soil management entry."

    def test_ipm_query(self, kb):
        results = kb.search("integrated pest management")
        assert len(results) > 0
        found_ipm = any(
            "Integrated Pest Management" in r["entry"]["data"].get("topic", "")
            for r in results
        )
        assert found_ipm, "Should return the IPM entry."

    def test_crop_insurance_query(self, kb):
        results = kb.search("crop insurance for farmers")
        assert len(results) > 0
        found_pmfby = any(
            "Fasal Bima" in r["entry"]["data"].get("name", "")
            for r in results
        )
        assert found_pmfby, "Should return PMFBY scheme entry."

    def test_type_filter(self, kb):
        results = kb.search("rice", entry_type="crop")
        for r in results:
            assert r["entry"]["type"] == "crop"

    def test_top_k_limit(self, kb):
        results = kb.search("disease management", top_k=2)
        assert len(results) <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANSWER ENGINE — TEMPLATE-BASED GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnswerEngine:
    """Test template-based answer generation."""

    def test_answer_has_required_keys(self, engine):
        result = engine.answer("How to grow rice?")
        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert "retrieved" in result
        assert "confidence" in result
        assert "method" in result

    def test_method_is_template(self, engine):
        result = engine.answer("How to grow rice?")
        assert result["method"] == "template"

    def test_rice_blast_answer_quality(self, engine):
        result = engine.answer("How to manage blast disease in rice?")
        answer = result["answer"]
        assert result["confidence"] in ("high", "medium")
        # Answer should mention key information
        assert "Blast" in answer or "blast" in answer
        assert "Magnaporthe" in answer or "Pyricularia" in answer or "Tricyclazole" in answer
        assert len(result["sources"]) > 0

    def test_wheat_fertilizer_answer(self, engine):
        result = engine.answer("What fertilizer should I use for wheat?")
        answer = result["answer"]
        assert result["confidence"] in ("high", "medium")
        assert "Wheat" in answer or "wheat" in answer

    def test_pm_kisan_answer(self, engine):
        result = engine.answer("Tell me about PM Kisan scheme")
        answer = result["answer"]
        assert "6,000" in answer or "6000" in answer or "₹6,000" in answer
        assert len(result["sources"]) > 0

    def test_fall_armyworm_answer(self, engine):
        result = engine.answer("How to control fall armyworm in maize?")
        answer = result["answer"]
        assert "Fall Armyworm" in answer or "armyworm" in answer.lower()
        assert "Management" in answer or "management" in answer.lower()

    def test_confidence_low_for_irrelevant_query(self, engine):
        result = engine.answer("How to build a spaceship?")
        assert result["confidence"] == "low"
        # Should give a helpful fallback message
        assert "Krishi Vigyan Kendra" in result["answer"] or "KVK" in result["answer"]

    def test_empty_question(self, engine):
        result = engine.answer("")
        assert result["confidence"] == "low"
        assert "valid" in result["answer"].lower() or "enter" in result["answer"].lower()

    def test_none_question(self, engine):
        result = engine.answer(None)
        assert result["confidence"] == "low"

    def test_sources_are_credible(self, engine):
        result = engine.answer("Rice blast disease")
        sources = result["sources"]
        # All sources should come from trusted institutions
        trusted_keywords = [
            "ICAR", "KVK", "IARI", "IIHR", "IIVR", "NRRI", "IIRR",
            "IIWBR", "CICR", "IIPR", "IIMR", "NBAIR", "ICRISAT",
            "TNAU", "Ministry", "NABARD", "University",
            "Directorate", "SBI Coimbatore", "CropLife", "CPRI",
            "IRRI", "CIMMYT", "Cotton Corporation",
        ]
        for src in sources:
            found = any(kw in src for kw in trusted_keywords)
            assert found, f"Source '{src}' does not appear to be from a trusted institution."


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EDGE CASES AND UNCERTAINTY HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test ambiguous, vague, and unsupported queries."""

    def test_vague_query_handled(self, engine):
        result = engine.answer("disease")
        # Should return something (may be medium confidence) but not crash
        assert result["answer"]
        assert result["confidence"] in ("high", "medium", "low")

    def test_nonsensical_query(self, engine):
        result = engine.answer("asdfghjkl zxcvbnm qwerty")
        assert result["confidence"] == "low"

    def test_out_of_domain_query(self, engine):
        result = engine.answer("What is the GDP of India?")
        # "India" appears in the KB so a medium match is acceptable, but
        # the answer should still include a caution or the system should
        # not claim high confidence for a non-agricultural question.
        assert result["confidence"] in ("low", "medium")
        if result["confidence"] == "low":
            assert "Krishi Vigyan Kendra" in result["answer"] or "KVK" in result["answer"]
        elif result["confidence"] == "medium":
            # Medium-confidence answers append an uncertainty note
            assert "Note" in result["answer"] or "KVK" in result["answer"]

    def test_multiple_crops_query(self, engine):
        result = engine.answer("rice and wheat")
        # Should return results for both
        assert result["answer"]
        assert len(result["retrieved"]) >= 1

    def test_hindi_transliterated_query(self, engine):
        result = engine.answer("chawal mein blast rog ka upchar")
        # 'chawal' is close to 'चावल' — may or may not match depending on
        # whether Hindi text is in the searchable content
        assert result["answer"]  # Should not crash

    def test_very_long_query(self, engine):
        long_q = "How to manage " + "blast disease " * 50 + "in rice?"
        result = engine.answer(long_q)
        assert result["answer"]  # Should handle gracefully

    def test_special_characters_in_query(self, engine):
        result = engine.answer("rice blast @#$%^&*() !!! ???")
        assert result["answer"]

    def test_uncertain_diagnosis_not_confirmed(self, engine):
        """
        When confidence is low, the system should NOT present uncertain
        information as confirmed fact.
        """
        result = engine.answer("My crop has some spots, what disease is it?")
        if result["confidence"] == "low":
            answer = result["answer"]
            assert (
                "consult" in answer.lower()
                or "KVK" in answer
                or "Krishi Vigyan Kendra" in answer
                or "uncertain" in answer.lower()
                or "could not find" in answer.lower()
            ), "Low-confidence answers should recommend consulting experts."


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANSWER VALIDATION AGAINST TRUSTED SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnswerValidation:
    """
    Validate that answers contain factually correct information
    cross-checked against the knowledge base.
    """

    def test_rice_blast_causal_organism(self, engine):
        """Rice blast is caused by Magnaporthe oryzae — answer should reflect this."""
        result = engine.answer("What causes blast disease in rice?")
        answer = result["answer"]
        assert (
            "Magnaporthe" in answer
            or "Pyricularia" in answer  # older name, also acceptable
        ), "Answer should mention the correct causal organism for rice blast."

    def test_wheat_sowing_months(self, engine):
        """Wheat is sown Oct-Dec in India."""
        result = engine.answer("When to sow wheat?")
        answer = result["answer"]
        assert (
            "October" in answer or "November" in answer
        ), "Answer should mention correct wheat sowing months."

    def test_chickpea_is_rabi(self, engine):
        """Chickpea is a rabi crop."""
        result = engine.answer("When is chickpea grown?")
        answer = result["answer"]
        assert "rabi" in answer.lower(), "Answer should identify chickpea as a rabi crop."

    def test_cotton_bollworm_management_mentions_bt(self, engine):
        """Bt cotton is the primary bollworm management strategy."""
        result = engine.answer("How to control American bollworm in cotton?")
        answer = result["answer"]
        assert (
            "Bt" in answer or "Bt cotton" in answer
        ), "Answer should mention Bt cotton for bollworm management."

    def test_pm_kisan_amount(self, engine):
        """PM-Kisan provides ₹6,000 per year."""
        result = engine.answer("How much money does PM Kisan give?")
        answer = result["answer"]
        assert (
            "6,000" in answer or "6000" in answer
        ), "Answer should mention ₹6,000 for PM-Kisan."

    def test_pmfby_premium_rates(self, engine):
        """PMFBY premium: 2% kharif, 1.5% rabi."""
        result = engine.answer("What is the premium for crop insurance?")
        answer = result["answer"]
        assert (
            "2%" in answer or "1.5%" in answer
        ), "Answer should mention correct PMFBY premium rates."

    def test_no_harmful_recommendation(self, engine):
        """Answers should not recommend banned or dangerous pesticides."""
        banned_pesticides = [
            "Endosulfan", "Monocrotophos", "Methyl Parathion",
            "Phorate", "Triazophos", "Dichlorvos",
        ]
        # Test across several queries
        queries = [
            "How to control pests in rice?",
            "How to manage bollworm in cotton?",
            "Pest management in maize",
        ]
        for q in queries:
            result = engine.answer(q)
            answer = result["answer"]
            for pesticide in banned_pesticides:
                assert pesticide not in answer, (
                    f"Answer for '{q}' should NOT recommend banned "
                    f"pesticide: {pesticide}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INTEGRATION TEST — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration test: question → retrieval → answer."""

    SAMPLE_QUESTIONS = [
        ("How to manage blast disease in rice?", "disease"),
        ("What fertilizer should I use for wheat?", "crop"),
        ("Tell me about PM Kisan scheme", "government_scheme"),
        ("Fall armyworm in maize", "pest"),
        ("How to irrigate sugarcane?", "crop"),
        ("What is soil testing?", "soil_management"),
        ("Integrated pest management", "general_practice"),
        ("Tomato leaf curl virus", "disease"),
        ("How to get crop insurance?", "government_scheme"),
        ("Fusarium wilt in chickpea", "disease"),
    ]

    @pytest.mark.parametrize("question, expected_type", SAMPLE_QUESTIONS)
    def test_sample_question(self, engine, question, expected_type):
        result = engine.answer(question)
        assert result["answer"], f"No answer generated for: {question}"
        assert result["confidence"] != "low", (
            f"Confidence too low for a supported question: {question}"
        )
        # Check that the expected entry type appears in retrieved entries
        retrieved_types = [r["type"] for r in result["retrieved"]]
        assert expected_type in retrieved_types, (
            f"Expected '{expected_type}' in retrieved types for '{question}', "
            f"got {retrieved_types}"
        )


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
