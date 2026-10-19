"""shared helpers: citation tags and cached IEEE text for the bibliography field (used by bib_v9.py and intro_v9.py)"""
import re, unicodedata
def ascii_(s): return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode())
def person(name):
    toks = name.split(); i = 0
    while i < len(toks) - 1 and re.fullmatch(r"([A-Z]\.-?)+", toks[i]): i += 1
    inits = toks[:i]; return " ".join(toks[i:]), (inits[0] if inits else ""), " ".join(inits[1:])
def make_tags(recs):
    tags, used = {}, set()
    for n, r in sorted(recs.items()):
        last = person(r["authors"][0])[0]; base = ascii_(last.split()[-1]) + r["year"][2:]
        tag = base; k = 2
        while tag in used: tag = f"{base}{k}"; k += 1
        used.add(tag); tags[n] = tag
    return tags
def fmt_authors(names):
    if len(names) > 6: return names[0] + " et al."
    if len(names) == 1: return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]
def entry_runs(recs, n, rpr, run):
    r = recs[n]; a = fmt_authors(r["authors"]); t = r["title"]; y = r["year"]; out = []
    def plain(s): out.append(run(s, rpr))
    def ital(s): out.append(run(s, rpr, italic=True))
    if r["type"] == "JournalArticle":
        plain(f'{a}, "{t}," '); ital(r["journal"] or "(journal)")
        tail = ""
        if r.get("volume"): tail += f", vol. {r['volume']}"
        if r.get("issue"): tail += f", no. {r['issue']}"
        if r.get("pages"): tail += (", p. " if "-" not in r["pages"] else ", pp. ") + r["pages"]
        plain(f"{tail}, {y}.")
    elif r["type"] == "Book":
        plain(f"{a}, "); ital(t); plain(f". {r.get('city','')}: {r.get('publisher','')}, {y}.")
    elif r["type"] == "BookSection":
        plain(f'{a}, "{t}," in '); ital(r["book"])
        plain(f", {fmt_authors(r.get('editors', []))}, Eds. {r.get('city','')}: {r.get('publisher','')}, {y}, pp. {r['pages']}.")
    elif r["type"] == "ConferenceProceedings":
        plain(f'{a}, "{t}," in '); ital(r["conference"])
        tail = (f", {r['city']}" if r.get("city") else "") + f", {y}" + (f", pp. {r['pages']}" if r.get("pages") else "")
        plain(tail + ".")
    elif r["type"] == "Report":
        plain(f'{a}, "{t}," {r.get("reporttype","")}, {r["institution"]}, {r["city"]}, {y}.'.replace(" ,", ","))
    else:
        plain(f'{a}, "{t}," {r.get("pubtitle","")}, {y}.')
    return out
