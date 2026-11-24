"""Tests for LLM commentary: facts, grounding checks, guardrail, and the endpoint. No network."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.commentary.facts import build_facts, template_commentary
from src.commentary.grounding import check_grounding
from src.commentary.ratelimit import RateLimiter


def _snap(**kw) -> dict:
    s = dict(attackers_alive=5, defenders_alive=5, spike_planted=0, time_elapsed_ms=0,
             time_remaining_ms=100000, plant_time_ms=0, attacker_economy=20000,
             defender_economy=20000, round_num=4, attacker_score=3, defender_score=1, plant_site="")
    s.update(kw)
    return s


# --- facts -----------------------------------------------------------------

def test_facts_events_and_numbers():
    start = _snap()
    kill = _snap(defenders_alive=4, time_elapsed_ms=20000, time_remaining_ms=80000)
    plant = _snap(defenders_alive=4, spike_planted=1, plant_site="B", time_elapsed_ms=40000,
                  time_remaining_ms=45000, plant_time_ms=40000)
    f = build_facts([start], [0.5])
    assert f["event"] == "round_start" and f["attacker_win_pct_change"] is None
    f = build_facts([start, kill], [0.5, 0.631])
    assert f["event"] == "defender_killed"
    assert (f["attacker_win_pct"], f["defender_win_pct"], f["attacker_win_pct_change"]) == (63, 37, 13)
    assert f["plant_site"] is None
    f = build_facts([start, kill, plant], [0.5, 0.63, 0.71])
    assert f["event"] == "spike_planted" and f["plant_site"] == "B" and f["seconds_left"] == 45
    assert f["round_number"] == 5


# --- grounding ---------------------------------------------------------------

PRE = build_facts([_snap(), _snap(defenders_alive=4, time_remaining_ms=80000)], [0.5, 0.63])
POST = build_facts([_snap(defenders_alive=4), _snap(defenders_alive=4, spike_planted=1, plant_site="B",
                                                     time_remaining_ms=45000, plant_time_ms=40000)],
                   [0.63, 0.71])


@pytest.mark.parametrize("line", [
    "Defender down! Attackers go up 5v4 and climb 13 points to 63%.",
    "That's a big pick, the defense now sits at 37% with numbers down.",
    "Spike down on B, attackers holding 71% in the 5v4.",
    "5 on 4 after the plant at B site, attackers are favorites at 71 percent.",
])
def test_grounded_lines_pass(line):
    facts = POST if ("B" in line or "plant" in line) else PRE
    assert check_grounding(line, facts) == []


@pytest.mark.parametrize("line, needle", [
    ("Attackers surge to 80% after that kill!", "percentage"),
    ("It's a 4v4 now and anything can happen.", "player count"),
    ("Spike planted, the defenders have to retake.", "isn't planted"),
    ("Big pressure building on A site.", "before any plant"),
    ("", "empty"),
    ("Line one.\nLine two.", "more than one line"),
])
def test_ungrounded_lines_fail(line, needle):
    issues = check_grounding(line, PRE)
    assert any(needle in i for i in issues), issues


def test_wrong_site_fails():
    assert any("plant is on B" in i for i in check_grounding("Spike is down on A, 71% for attack.", POST))


def test_article_a_is_not_site_a():
    assert check_grounding("What a site take by the attackers, 63% now.", PRE) == []


def test_template_is_always_grounded():
    for facts in (PRE, POST, build_facts([_snap()], [0.48])):
        assert check_grounding(template_commentary(facts), facts) == []


# --- guardrail ---------------------------------------------------------------

def _fake_client(text: str | None, stop_reason: str = "end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    response = SimpleNamespace(content=content, stop_reason=stop_reason,
                               usage=SimpleNamespace(input_tokens=300, output_tokens=40))
    create = lambda **kwargs: response  # noqa: E731
    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


def test_guardrail_keeps_grounded_llm_line():
    from src.commentary.generate import grounded_commentary
    out = grounded_commentary(PRE, _fake_client("Defender down, attackers up 5v4 at 63%."))
    assert out == {"commentary": "Defender down, attackers up 5v4 at 63%.", "source": "llm", "rejected": []}


def test_guardrail_replaces_hallucinated_line():
    from src.commentary.generate import grounded_commentary
    out = grounded_commentary(PRE, _fake_client("TenZ with the Operator pick, attackers at 90%!"))
    assert out["source"] == "template"
    assert out["commentary"] == template_commentary(PRE)
    assert out["rejected"]


def test_guardrail_handles_refusal():
    from src.commentary.generate import grounded_commentary
    out = grounded_commentary(PRE, _fake_client(None, stop_reason="refusal"))
    assert out["source"] == "template"


# --- rate limiter ------------------------------------------------------------

def test_rate_limiter_per_minute_and_daily_cap():
    now = [0.0]
    rl = RateLimiter(per_minute=2, daily_cap=3, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("a") and not rl.allow("a")   # per-minute limit
    now[0] = 61
    assert rl.allow("a")                                             # window reset; 3 used today
    assert not rl.allow("b")                                         # daily cap hit for everyone


# --- endpoint ----------------------------------------------------------------

@pytest.fixture
def serve_module():
    if not Path("artifacts/models/best_weights.npz").exists():
        pytest.skip("weights not present")
    from fastapi.testclient import TestClient
    from scripts import serve
    return serve, TestClient(serve.app)


def test_commentary_disabled_without_key(serve_module, monkeypatch):
    serve, client = serve_module
    monkeypatch.setattr(serve, "COMMENTARY_ENABLED", False)
    assert client.get("/health").json()["commentary"] is False
    assert client.post("/commentary", json={"snapshots": [_snap()]}).status_code == 503


def test_commentary_endpoint_with_fake_client(serve_module, monkeypatch):
    serve, client = serve_module
    monkeypatch.setattr(serve, "COMMENTARY_ENABLED", True)
    monkeypatch.setattr(serve, "commentary_limiter", RateLimiter(per_minute=1, daily_cap=10))
    # The fake echoes a fixed line; the endpoint must reject it if the numbers don't match.
    monkeypatch.setattr(serve, "commentary_client", _fake_client("Attackers at 99% already!"))
    body = {"snapshots": [_snap(), _snap(defenders_alive=4, time_remaining_ms=80000)]}
    r = client.post("/commentary", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "template" and data["rejected"]
    assert client.post("/commentary", json=body).status_code == 429  # per-minute limit of 1


# --- score and kill-location checks (from the first eval's failures) ----------

def test_projected_score_fails():
    # PRE has score 3-1
    assert any("isn't the current score" in i for i in check_grounding(
        "Attackers up 5v4 at 63%, one round from closing this out at 4-1!", PRE))


def test_current_score_passes_either_order():
    assert check_grounding("At 3-1, attackers up 5v4 and sitting at 63%.", PRE) == []
    assert check_grounding("Defenders trail 1-3 and now sit at 37%.", PRE) == []


def test_kill_location_fails_even_on_plant_site():
    issues = check_grounding("Defender down on B, attackers at 71% in the 5v4.", POST)
    assert any("where a kill happened" in i for i in issues)


def test_spike_on_site_still_passes():
    assert check_grounding("Spike down on B, attackers holding 71% in the 5v4.", POST) == []


@pytest.mark.parametrize("line", [
    "Another defender drops, it's 4v2 and attackers surge to 96%, one round from closing this out at 11-2!",
    "Spike down on A, attackers at 87 percent, up 3 to level this at 4-4!",
])
def test_first_eval_failures_now_caught(line):
    """Lines the judge flagged in the baseline eval (scores 10-2 and 3-4 at the time)."""
    facts = dict(PRE, attacker_score=10, defender_score=2) if "11-2" in line else \
        dict(POST, attacker_score=3, defender_score=4, plant_site="A", attacker_win_pct=87,
             defender_win_pct=13, attacker_win_pct_change=3)
    assert check_grounding(line, facts)
