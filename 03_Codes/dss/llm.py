"""Which language model answers stage 3, and how it is reached.

Stage 3 never uses an API key. It runs on a chat subscription through the
vendor's own command-line tool:

    provider  CLI                     login                    account
    claude    Claude Code (`claude`)  claude-login.bat         Claude Pro/Max
    codex     OpenAI Codex (`codex`)  codex-login.bat          ChatGPT Plus/Pro

Selection (environment, read on every call, so the dashboard can switch
at run time):

    DSS_GENAI_PROVIDER   claude | codex                (default claude)
    DSS_GENAI_MODEL      model of that provider        (empty: plan default)
                         claude: opus, sonnet, haiku, claude-opus-5, ...
                         codex:  gpt-5.5, gpt-6-sol, ... (see list_models)
    DSS_GENAI_EFFORT     reasoning effort              (empty: CLI default)
                         claude: low|medium|high|xhigh|max
                         codex:  low|medium|high|xhigh|max|ultra
    DSS_GENAI_FALLBACK   provider[:model] to switch to when the selected
                         provider reports a usage or rate limit, e.g.
                         "codex:gpt-5.5" or "claude:sonnet". Empty: no switch.
    DSS_GENAI_CALL_LOG   JSONL file that receives one line per model call
                         (provider, model, effort, ok, latency, error).
                         Batch runs (DSS_REQUIRE_GENAI) default to
                         genai_calls.jsonl in the working directory.

The batch scripts read the same variables from 03_Codes/llm.cfg through
llm_env.bat; `python -m dss.llm select` (llm-select.bat) writes that file.

The Claude command line itself is built by adapt.genai_cmd, which predates
this module; this module adds the Codex transport, the provider switch, the
fallback and the call log.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

PROVIDERS = ("claude", "codex")

PROVIDER_LABEL = {
    "claude": "Claude (Claude Code CLI, Claude subscription)",
    "codex": "ChatGPT (OpenAI Codex CLI, ChatGPT subscription)",
}

#: shown when the CLI cannot be asked for its catalog
CLAUDE_MODELS = ["opus", "sonnet", "haiku", "claude-opus-5", "claude-sonnet-5"]
CODEX_MODELS_FALLBACK = ["gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra",
                         "gpt-5.6-luna", "gpt-6-sol", "gpt-6-luna",
                         "gpt-6.1-sol", "gpt-6.1-terra", "gpt-6-astra"]
EFFORTS = {
    "claude": ["low", "medium", "high", "xhigh", "max"],
    "codex": ["low", "medium", "high", "xhigh", "max", "ultra"],
}

#: same instruction the Claude path passes as --system-prompt. Codex has no
#: system-prompt flag, so it is placed in front of the opening prompt.
SYSTEM_TEXT = ("You emit exactly one minified JSON object and nothing else. "
               "No prose, no explanation, no markdown fence, no code block, "
               "no leading or trailing text. Any character outside the JSON "
               "object is discarded by the caller, so writing it only costs "
               "time. Keep every string field short. Do not run commands, "
               "read files or search the web; answer from the text given.")

# text that marks a usage or rate limit in either CLI's error output
_LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "quota",
                  "limit reached", "limit has been reached", "hit your limit",
                  "too many requests", "429", "credit balance",
                  "insufficient_quota", "usage_limit", "try again at",
                  "resets at", "upgrade to")

# provider/model switched to after a limit, for the rest of this process
_ACTIVE_OVERRIDE: dict = {}


# ------------------------------------------------------------ selection
def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def selected_provider() -> str:
    p = _env("DSS_GENAI_PROVIDER").lower() or "claude"
    if p in ("openai", "chatgpt", "gpt"):
        p = "codex"
    if p in ("anthropic",):
        p = "claude"
    return p if p in PROVIDERS else "claude"


def provider() -> str:
    """The provider the next call goes to (the fallback, once triggered)."""
    return _ACTIVE_OVERRIDE.get("provider") or selected_provider()


def _model_fits(prov: str, model: str) -> bool:
    m = model.lower()
    if prov == "claude":
        return not m.startswith(("gpt", "o1", "o3", "o4", "codex"))
    return not (m.startswith("claude") or m in ("opus", "sonnet", "haiku"))


def model(prov: Optional[str] = None) -> str:
    """Model passed to the CLI, or '' for the plan default. A name that
    belongs to the other provider is dropped rather than sent."""
    prov = prov or provider()
    if _ACTIVE_OVERRIDE.get("provider") == prov:
        m = _ACTIVE_OVERRIDE.get("model", "")
    else:
        m = _env("DSS_GENAI_MODEL")
    return m if m and _model_fits(prov, m) else ""


def effort(prov: Optional[str] = None) -> str:
    prov = prov or provider()
    e = _env("DSS_GENAI_EFFORT").lower()
    return e if e in EFFORTS.get(prov, []) else ""


def describe() -> str:
    """One line for logs: provider, model and effort actually in use."""
    p = provider()
    s = f"{p}:{model(p) or '(plan default)'}"
    if effort(p):
        s += f" effort {effort(p)}"
    if _ACTIVE_OVERRIDE:
        s += " [fallback]"
    return s


def is_limit_error(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in _LIMIT_MARKERS)


def try_fallback(err: str) -> bool:
    """Switch to DSS_GENAI_FALLBACK when `err` is a usage or rate limit.
    Returns True when the next call goes to a different provider or model.
    The switch lasts for the rest of the process and is printed and logged,
    so a campaign that changed model midway can be found and rerun."""
    fb = _env("DSS_GENAI_FALLBACK")
    if not fb or _ACTIVE_OVERRIDE or not is_limit_error(err):
        return False
    prov, _, mdl = fb.partition(":")
    prov = prov.strip().lower()
    prov = "codex" if prov in ("openai", "chatgpt", "gpt") else prov
    if prov not in PROVIDERS:
        return False
    if prov == selected_provider() and mdl.strip() == _env("DSS_GENAI_MODEL"):
        return False
    _ACTIVE_OVERRIDE.update(provider=prov, model=mdl.strip(),
                            since=time.strftime("%Y-%m-%d %H:%M:%S"),
                            reason=(err or "")[:200])
    print(f"[genai] FALLBACK: {selected_provider()} reported a limit "
          f"({(err or '')[:120]!r}); switching to {describe()}", flush=True)
    log_call(dict(event="fallback", to=describe(), reason=(err or "")[:300]))
    return True


def reset_fallback() -> None:
    _ACTIVE_OVERRIDE.clear()


def fallback_state() -> dict:
    return dict(_ACTIVE_OVERRIDE)


# ------------------------------------------------------------ call log
def _log_path() -> str:
    p = _env("DSS_GENAI_CALL_LOG")
    if p:
        return p
    if _env("DSS_REQUIRE_GENAI"):
        return os.path.join(os.getcwd(), "genai_calls.jsonl")
    return ""


def log_call(rec: dict) -> None:
    path = _log_path()
    if not path:
        return
    rec = dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid(), **rec)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


# ------------------------------------------------------------ codex CLI
_CODEX_LAUNCH: list = []


def codex_launch() -> list:
    """Executable part of the `codex` command. On Windows npm installs a
    `codex.cmd` shim that subprocess cannot start without cmd.exe, so the
    package's own entry script is handed to node directly (same reasoning
    as adapt.claude_launch). DSS_CODEX_EXE overrides the lookup."""
    if _CODEX_LAUNCH:
        return list(_CODEX_LAUNCH)
    exe = _env("DSS_CODEX_EXE") or shutil.which("codex") or "codex"
    launch = [exe]
    if exe.lower().endswith((".cmd", ".bat", ".ps1")):
        here = os.path.dirname(exe)
        node = shutil.which("node")
        if not node:
            for cand in (os.path.join(here, "node.exe"),
                         os.path.join(os.environ.get("ProgramFiles", ""),
                                      "nodejs", "node.exe")):
                if cand and os.path.exists(cand):
                    node = cand
                    break
        js = os.path.join(here, "node_modules", "@openai", "codex", "bin",
                          "codex.js")
        if node and os.path.exists(js):
            launch = [node, js]
        else:
            # cmd.exe can start the shim; the prompt travels on stdin anyway
            launch = [os.environ.get("COMSPEC", "cmd.exe"), "/c", exe]
    _CODEX_LAUNCH[:] = launch
    return list(launch)


def codex_available() -> bool:
    return bool(_env("DSS_CODEX_EXE") or shutil.which("codex"))


def claude_available() -> bool:
    return bool(_env("DSS_CLAUDE_EXE") or shutil.which("claude"))


def available(prov: Optional[str] = None) -> bool:
    prov = prov or provider()
    return codex_available() if prov == "codex" else claude_available()


def _codex_workdir() -> str:
    """An empty directory the Codex agent runs in, so a stray command has
    nothing to read. Kept stable so `codex exec resume` finds the thread."""
    base = _env("CODEX_HOME") or tempfile.gettempdir()
    d = os.path.join(base, "dss-work")
    os.makedirs(d, exist_ok=True)
    return d


def codex_cmd(session: Optional[str] = None, resume: bool = False,
              out_file: str = "") -> list:
    cmd = codex_launch() + ["exec"]
    if session and resume:
        cmd += ["resume", session]
    cmd += ["--skip-git-repo-check", "--json"]
    if not resume:
        cmd += ["-s", "read-only", "--color", "never",
                "-C", _codex_workdir()]
    m = model("codex")
    if m:
        cmd += ["-m", m]
    e = effort("codex")
    if e:
        cmd += ["-c", f'model_reasoning_effort="{e}"']
    if out_file:
        cmd += ["-o", out_file]
    cmd += ["-"]                       # prompt on stdin
    return cmd


def run_codex(prompt: str, timeout: float, session: Optional[str] = None,
              resume: bool = False) -> dict:
    """One `codex exec` call. Returns dict(rc, text, err, session, usage,
    cmd, raw). `session` is the Codex thread id of an earlier call; a fresh
    call creates a thread and returns its id."""
    fd, out_file = tempfile.mkstemp(prefix="dss_codex_", suffix=".txt")
    os.close(fd)
    cmd = codex_cmd(session, resume, out_file)
    full = prompt if resume else SYSTEM_TEXT + "\n\n" + prompt
    out = dict(rc=None, text="", err="", session=session or "", usage=None,
               cmd=" ".join(cmd), raw="")
    try:
        res = subprocess.run(cmd, input=full, capture_output=True, text=True,
                             encoding="utf-8", errors="replace",
                             timeout=timeout, cwd=_codex_workdir())
    except subprocess.TimeoutExpired:
        out["err"] = f"timeout after {timeout:.0f} s"
        out["rc"] = "timeout"
        _rm(out_file)
        return out
    except Exception as exc:
        out["err"] = f"launch failed ({type(exc).__name__})"
        out["rc"] = "launch"
        _rm(out_file)
        return out
    out["rc"] = res.returncode
    out["raw"] = (res.stdout or "")[-4000:]
    last_err, msgs = "", []
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        typ = ev.get("type", "")
        if typ == "thread.started" and ev.get("thread_id"):
            out["session"] = ev["thread_id"]
        elif typ == "error":
            last_err = str(ev.get("message", ""))
        elif typ == "turn.failed":
            last_err = str((ev.get("error") or {}).get("message", "")
                           or last_err or "turn failed")
        elif typ == "turn.completed":
            out["usage"] = ev.get("usage")
        elif typ == "item.completed":
            it = ev.get("item") or {}
            if it.get("type") in ("agent_message", "assistant_message"):
                msgs.append(str(it.get("text", "")))
    try:
        with open(out_file, encoding="utf-8", errors="replace") as fh:
            text = fh.read().strip()
    except OSError:
        text = ""
    _rm(out_file)
    out["text"] = text or (msgs[-1].strip() if msgs else "")
    if res.returncode != 0 or not out["text"]:
        out["err"] = (last_err or (res.stderr or "").strip()[-300:]
                      or f"codex exit {res.returncode}, no reply")
        if res.returncode == 0:
            out["rc"] = 1
    return out


def _rm(p: str) -> None:
    try:
        os.remove(p)
    except OSError:
        pass


# ------------------------------------------------------------ catalog
def list_models(prov: str) -> list:
    """Model names offered in the dashboard for one provider."""
    if prov == "claude":
        return list(CLAUDE_MODELS)
    if not codex_available():
        return list(CODEX_MODELS_FALLBACK)
    try:
        res = subprocess.run(codex_launch() + ["debug", "models"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=20,
                             stdin=subprocess.DEVNULL)
        data = json.loads(res.stdout)
        rows = data.get("models", data) if isinstance(data, dict) else data
        names = [r["slug"] for r in rows
                 if isinstance(r, dict) and r.get("slug")
                 and r.get("visibility", "list") == "list"]
        return names or list(CODEX_MODELS_FALLBACK)
    except Exception:
        return list(CODEX_MODELS_FALLBACK)


def login_status(prov: str) -> dict:
    """dict(installed, logged_in, text) from the CLI's own status command."""
    if not available(prov):
        return dict(installed=False, logged_in=False,
                    text=("`codex` not found. Install Node.js, then run: "
                          "npm i -g @openai/codex" if prov == "codex" else
                          "`claude` not found. Install Claude Code."))
    if prov == "codex":
        cmd = codex_launch() + ["login", "status"]
    else:
        from .adapt import claude_launch
        cmd = claude_launch() + ["auth", "status", "--text"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30,
                             stdin=subprocess.DEVNULL)
        txt = ((res.stdout or "") + (res.stderr or "")).strip()
        lines = [l for l in txt.splitlines()
                 if not l.startswith("WARNING: proceeding")]
        txt = "\n".join(lines).strip()
        ok = res.returncode == 0 and "not logged in" not in txt.lower()
        return dict(installed=True, logged_in=ok, text=txt[:600])
    except Exception as exc:
        return dict(installed=True, logged_in=False,
                    text=f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ llm.cfg
CFG_KEYS = ("DSS_GENAI_PROVIDER", "DSS_GENAI_MODEL", "DSS_GENAI_EFFORT",
            "DSS_GENAI_FALLBACK")


def cfg_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "llm.cfg")


def read_cfg() -> dict:
    out = {}
    try:
        with open(cfg_path(), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                if k.strip() in CFG_KEYS:
                    out[k.strip()] = v.strip()
    except OSError:
        pass
    return out


def write_cfg(values: dict) -> str:
    lines = ["# Language model for the batch scripts (read by llm_env.bat).",
             "# Written by llm-select.bat or the dashboard. One run campaign",
             "# should use one provider and one model from start to end."]
    for k in CFG_KEYS:
        lines.append(f"{k}={(values.get(k) or '').strip()}")
    with open(cfg_path(), "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return cfg_path()


def _ask(title: str, options: list, default: str = "") -> str:
    print(f"\n{title}")
    for i, o in enumerate(options, 1):
        print(f"  {i}) {o}")
    print("  or type a name; Enter keeps", repr(default or "(empty)"))
    a = input("> ").strip()
    if not a:
        return default
    if a.isdigit() and 1 <= int(a) <= len(options):
        v = options[int(a) - 1]
        return "" if v.startswith("(") else v
    return a


def _select() -> None:
    cur = read_cfg()
    prov = _ask("Provider", ["claude", "codex"],
                cur.get("DSS_GENAI_PROVIDER", "claude"))
    prov = "codex" if prov in ("openai", "chatgpt", "gpt") else prov
    st = login_status(prov)
    print(f"[{prov}] {st['text']}")
    if not st["logged_in"]:
        print(f"Not logged in: run {prov}-login.bat before the campaign.")
    keep = cur.get("DSS_GENAI_MODEL", "")
    if keep and not _model_fits(prov, keep):
        keep = ""
    mdl = _ask("Model", ["(plan default)"] + list_models(prov), keep)
    eff = _ask("Reasoning effort", ["(CLI default)"] + EFFORTS[prov],
               cur.get("DSS_GENAI_EFFORT", ""))
    other = "codex" if prov == "claude" else "claude"
    fb = _ask("Fallback when the limit is hit",
              ["(none)", f"{other}:{list_models(other)[0]}"],
              cur.get("DSS_GENAI_FALLBACK", ""))
    p = write_cfg(dict(DSS_GENAI_PROVIDER=prov, DSS_GENAI_MODEL=mdl,
                       DSS_GENAI_EFFORT=eff, DSS_GENAI_FALLBACK=fb))
    print(f"\nSaved {p}:")
    print(open(p, encoding="utf-8").read())


def _status() -> None:
    print("selected:", describe())
    print("llm.cfg :", read_cfg() or "(none)")
    for p in PROVIDERS:
        st = login_status(p)
        print(f"\n[{p}] installed={st['installed']} "
              f"logged_in={st['logged_in']}\n{st['text']}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "status"
    if arg == "select":
        _select()
    elif arg == "models":
        for p in PROVIDERS:
            print(p, ", ".join(list_models(p)))
    else:
        _status()
