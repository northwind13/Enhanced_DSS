"""Stop a campaign step that recorded a new-concept proposal.

    python check_noconcept.py <out_dir> [--model] [--no-mutation]

Exit code 1 when any proposal in <out_dir>/proposals.jsonl carries a new
intermediate concept (DSS_NO_CONCEPT=1 must have prevented it). With
--model the step must also hold proposals written by the language model,
and calls that found the model unreachable are counted. With --no-mutation
every admitted object must have been written by the model itself (the
screened proposer with DSS_LLM_SCREEN_FILL=0)."""
import json, os, sys

d = sys.argv[1]
need_model = "--model" in sys.argv
no_mut = "--no-mutation" in sys.argv
p = os.path.join(d, "proposals.jsonl")
if not os.path.exists(p):
    print(f"check: {p} missing"); sys.exit(1)
n = conc = model = down = 0
origin = {}
with open(p, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        if r.get("kind") == "concept" or r.get("concept"):
            conc += 1
        if r.get("source") in ("llm", "llmscreen"):
            model += 1
        if "not reachable" in str(r.get("detail") or ""):
            down += 1
        if r.get("accepted"):
            o = str((r.get("search") or {}).get("origin") or r.get("source"))
            origin[o] = origin.get(o, 0) + 1
print(f"check {d}: proposals {n}, new concepts {conc}, from the model {model}, "
      f"model unreachable {down}, admitted by origin {origin}")
if conc:
    print("STOP: new-concept proposals found although DSS_NO_CONCEPT=1"); sys.exit(1)
if need_model and model == 0:
    print("STOP: no proposal came from the model"); sys.exit(1)
if no_mut and origin.get("mutation", 0):
    print("STOP: admitted variations found although DSS_LLM_SCREEN_FILL=0"); sys.exit(1)
sys.exit(0)
