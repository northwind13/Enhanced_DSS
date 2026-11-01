"""Provider switch of stage 3 (dss.llm): Claude Code and OpenAI Codex CLIs,
one session per incident on each, and the fallback after a usage limit.
The CLIs are replaced by small fake scripts, so no account is needed."""
import json
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt",
                                reason="fake CLIs are POSIX scripts")

FAKE_CODEX = r'''#!PY
import sys, json, uuid
a = sys.argv[1:]
sys.stdin.read()
tid = a[a.index("resume") + 1] if "resume" in a else str(uuid.uuid4())
print(json.dumps({"type": "thread.started", "thread_id": tid}))
open(a[a.index("-o") + 1], "w").write(
    '{"rule":"x","resumed":%s}' % ("true" if "resume" in a else "false"))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}))
'''

FAKE_CLAUDE = r'''#!PY
import sys, json, os
if os.environ.get("FAKE_CLAUDE_LIMIT"):
    print(json.dumps({"is_error": True,
                      "result": "Claude AI usage limit reached"}))
    sys.exit(1)
print(json.dumps({"subtype": "success", "result": '{"rule":"c"}'}))
'''


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    from dss import adapt, llm
    for name, src in (("codex", FAKE_CODEX), ("claude", FAKE_CLAUDE)):
        p = tmp_path / name
        p.write_text(src.replace("PY", sys.executable))
        p.chmod(0o755)
    monkeypatch.setenv("DSS_CODEX_EXE", str(tmp_path / "codex"))
    monkeypatch.setenv("DSS_CLAUDE_EXE", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "home"))
    for k in ("DSS_GENAI_PROVIDER", "DSS_GENAI_MODEL", "DSS_GENAI_EFFORT",
              "DSS_GENAI_FALLBACK", "DSS_REQUIRE_GENAI", "FAKE_CLAUDE_LIMIT"):
        monkeypatch.delenv(k, raising=False)
    adapt._CLAUDE_LAUNCH.clear()
    llm._CODEX_LAUNCH.clear()
    llm.reset_fallback()
    yield adapt, llm
    adapt._CLAUDE_LAUNCH.clear()
    llm._CODEX_LAUNCH.clear()
    llm.reset_fallback()


class _Eng:
    pass


def test_codex_session_and_provenance(fakes, monkeypatch):
    adapt, llm = fakes
    monkeypatch.setenv("DSS_GENAI_PROVIDER", "codex")
    monkeypatch.setenv("DSS_GENAI_MODEL", "gpt-x")
    monkeypatch.setenv("DSS_GENAI_EFFORT", "high")
    assert adapt.genai_config()["provider"] == "codex"
    e = _Eng()
    p1 = adapt._genai_propose("s", 20, engine=e)
    assert p1["resumed"] is False and p1["_llm"] == "codex:gpt-x"
    p2 = adapt._genai_propose("s2", 20, engine=e)
    assert p2["resumed"] is True              # same thread resumed
    assert "-c" in llm.codex_cmd() and "-m" in llm.codex_cmd()


def test_claude_model_not_sent_to_codex(fakes, monkeypatch):
    adapt, llm = fakes
    monkeypatch.setenv("DSS_GENAI_PROVIDER", "codex")
    monkeypatch.setenv("DSS_GENAI_MODEL", "opus")
    assert llm.model() == "" and "-m" not in llm.codex_cmd()


def test_fallback_on_limit(fakes, monkeypatch, tmp_path):
    adapt, llm = fakes
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("DSS_GENAI_CALL_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_LIMIT", "1")
    monkeypatch.setenv("DSS_GENAI_FALLBACK", "codex:gpt-x")
    e = _Eng()
    p = adapt._genai_propose("s", 20, engine=e)
    assert p["_llm"] == "codex:gpt-x" and llm.provider() == "codex"
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert rows[0]["provider"] == "claude" and not rows[0]["ok"]
    assert rows[1]["event"] == "fallback" and rows[2]["ok"]


def test_no_fallback_without_setting(fakes, monkeypatch):
    adapt, llm = fakes
    monkeypatch.setenv("FAKE_CLAUDE_LIMIT", "1")
    assert adapt._genai_propose("s", 20, engine=_Eng()) is None
    assert llm.provider() == "claude"
