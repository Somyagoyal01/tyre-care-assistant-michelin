import pytest

import run_eval
from assistant.agent import TyreAssistant
from assistant.retriever import Retriever
from assistant.router import classify
from assistant.tyre_size import TyreSizeError, catalogue_matches, check_fitment, parse_size


@pytest.fixture(scope="module")
def bot():
    return TyreAssistant()


def test_size_parser_variants():
    s = parse_size("205/55 R16 91V")
    assert (s.width_mm, s.aspect_ratio, s.construction, s.rim_in) == (205, 55, "R", 16)
    assert (s.load_index, s.speed_symbol) == (91, "V")
    s2 = parse_size("my tyre is 225/45ZR17")
    assert (s2.construction, s2.rim_in, s2.load_index) == ("ZR", 17, None)
    with pytest.raises(TyreSizeError):
        parse_size("no size here")


def test_load_and_speed_decode():
    s = parse_size("195/65 R15 91H")
    assert s.load_kg == 615 and s.speed_kmh == 210
    assert parse_size("225/45 R17 94W").load_kg == 670
    assert "615 kg" in s.describe() and "210 km/h" in s.describe()


def test_fitment_check():
    ok = check_fitment(parse_size("225/45 R17 94W"), 91, "V")
    assert ok.fits
    bad = check_fitment(parse_size("205/55 R16 88H"), 91, "V")
    assert not bad.fits and any("LOWER" in r for r in bad.reasons)
    rim = check_fitment(parse_size("225/45 R17 94W"), 91, "V", parse_size("205/55 R16 91V"))
    assert not rim.fits  # different rim diameter


def test_catalogue_lookup():
    skus = [r["sku"] for r in catalogue_matches(parse_size("205/55 R16 91V"))]
    assert skus == ["SYN-001", "SYN-002"]  # 91H is excluded (H < V)


def test_router_safety_refusal(bot):
    assert classify("There is a bulge on my sidewall, can I drive to work?") == "safety"
    r = bot.ask("My tyre has a bubble on the side, is it safe to drive?")
    assert r["route"] == "safety"
    assert "do not drive" in r["answer"].lower() and "safe to drive" not in r["answer"].lower().replace(
        "confirm a damaged tyre is safe", "")
    assert bot.ask("What is the price of a new tyre?")["route"] == "out_of_scope"


def test_retrieval_known_query():
    top = [d.doc_id for d, _ in Retriever().search("what is the minimum legal tread depth", k=3)]
    assert top[0] == "tread_depth"


def test_answers_cite_docs(bot):
    r = bot.ask("How often should I rotate my tyres?")
    assert r["route"] == "rag" and "rotation" in r["doc_ids"] and "Sources: rotation" in r["answer"]


def test_empty_question_handled(bot):
    assert bot.ask("   ")["route"] == "error"


def test_eval_harness_end_to_end(tmp_path):
    metrics = run_eval.main(["--split", "dev", "--report", str(tmp_path / "r.md")])
    assert metrics["dev"]["n"] >= 25
    assert metrics["dev"]["route_accuracy"] >= 0.9
    assert (tmp_path / "r.md").read_text().startswith("# Evaluation report")


def test_api_ask():
    from fastapi.testclient import TestClient

    import app
    c = TestClient(app.app)
    r = c.post("/ask", json={"question": "What does 205/55 R16 91V mean?"})
    assert r.status_code == 200 and r.json()["route"] == "tool"
    assert c.post("/ask", json={"question": ""}).status_code == 422


def test_required_size_first_is_not_reversed():
    """Regression for held-out h10: required size mentioned first, cheaper speed rating offered."""
    from assistant.agent import TyreAssistant
    out = TyreAssistant().ask("Car manual says 195/65 R15 91H. Shop is offering 195/65 R15 91T, is that fine?")
    assert out["tool_result"]["fits"] is False
    assert "LOWER" in out["answer"]
