"""Structured source records for the 52 references of DisasterAware_IEEESystems_v9 (order = first citation).
Each record: type, authors (list of 'First Middle Last' strings, initials allowed), title, and type specific fields.
Journal articles are parsed from refs_v9.txt; the rest and the incomplete ones are given explicitly."""
import re

TXT = "refs_v9.txt"


def split_authors(s):
    s = s.replace(", and ", ", ").replace(" and ", ", ")
    return [a.strip() for a in s.split(", ") if a.strip()]


def parse_journal(line):
    m = re.match(r'(?P<auth>.+?), "(?P<title>.+?),"\s*(?P<rest>.+)$', line)
    auth, title, rest = m.group("auth"), m.group("title"), m.group("rest")
    rec = {"type": "JournalArticle", "authors": split_authors(auth), "title": title}
    y = re.search(r"(\d{4})[a-z]?\.?$", rest); rec["year"] = y.group(1)
    v = re.search(r"vol\. (\d+)", rest); rec["volume"] = v.group(1) if v else ""
    n = re.search(r"no\. ([\d\-]+)", rest); rec["issue"] = n.group(1) if n else ""
    p = re.search(r"(?:pp\.|art\.) ([\w\-]+)", rest); rec["pages"] = p.group(1).replace("--", "-") if p else ""
    j = re.split(r", (?:vol\.|no\.|pp\.|art\.|\d{4})", rest)[0].strip()
    rec["journal"] = "" if re.match(r"^(vol\.|pp\.|art\.|\d{4})", j) else j
    return rec


OVERRIDE = {
    4: {"type": "Book", "authors": ["D. J. Power"], "title": "Decision Support Systems: Concepts and Resources for Managers",
        "year": "2002", "publisher": "Quorum Books", "city": "Westport, CT, USA"},
    5: {"type": "JournalArticle", "authors": ["F. Vásquez", "A. Cravero", "M. Castillo", "F. Riquelme"],
        "title": "Decision support system development of wildland fire: A systematic mapping", "journal": "Forests",
        "year": "2021", "volume": "12", "issue": "7", "pages": "943"},
    6: {"type": "Report", "authors": ["R. C. Rothermel"], "title": "A mathematical model for predicting fire spread in wildland fuels",
        "year": "1972", "institution": "USDA Forest Service, Intermountain Forest and Range Experiment Station", "city": "Ogden, UT, USA",
        "reporttype": "Res. Pap. INT-115"},
    7: {"type": "ConferenceProceedings", "authors": ["M. A. Finney"], "title": "An overview of FlamMap fire modeling capabilities",
        "conference": "Fuels Management: How to Measure Success, Conference Proceedings (RMRS-P-41)", "year": "2006", "pages": "213-220",
        "publisher": "USDA Forest Service", "city": "Portland, OR, USA"},
    8: {"type": "JournalArticle", "authors": ["C. Pais", "J. Carrasco", "P. E. Moudio", "Z.-J. M. Shen"],
        "title": "Downstream protection value: Detecting critical zones for effective fuel-treatment under wildfire risk",
        "journal": "", "year": "2021", "volume": "90", "pages": "101703"},
    9: {"type": "JournalArticle", "authors": ["T. Orphanoudakis", "C. Betzelos", "H. C. Leligou"],
        "title": "MORA: A multicriteria optimal resource allocation and decision support toolkit for wildfire management",
        "journal": "Algorithms", "year": "2025", "volume": "18", "issue": "11", "pages": "677"},
    10: {"type": "JournalArticle", "authors": ["E. K. Noonan-Wright", "T. S. Opperman", "M. A. Finney", "G. T. Zimmerman", "R. C. Seli",
                                                "L. M. Elenz", "D. E. Calkin", "J. R. Fiedler"],
         "title": "Developing the US wildland fire decision support system", "journal": "Journal of Combustion", "year": "2011",
         "volume": "2011", "pages": "168473"},
    11: {"type": "JournalArticle", "authors": ["S. D. Fillmore", "T. B. Paveglio"],
         "title": "Use of the Wildland Fire Decision Support System (WFDSS) for full suppression and managed fires within the Southwestern Region of the US Forest Service",
         "journal": "International Journal of Wildland Fire", "year": "2023", "volume": "32", "issue": "4", "pages": "622-635"},
    12: {"type": "JournalArticle", "authors": ["D. E. Calkin", "C. D. O'Connor", "M. P. Thompson", "R. D. Stratton"],
         "title": "Strategic wildfire response decision support and the risk management assistance program", "journal": "Forests",
         "year": "2021", "volume": "12", "issue": "10", "pages": "1407"},
    13: {"type": "JournalArticle", "authors": ["C. Rapp", "E. Rabung", "R. Wilson", "E. Toman"],
         "title": "Wildfire decision support tools: An exploratory study of use in the United States",
         "journal": "International Journal of Wildland Fire", "year": "2020", "volume": "29", "issue": "7", "pages": "581-594"},
    15: {"type": "Book", "authors": ["J. Malczewski"], "title": "GIS and Multicriteria Decision Analysis", "year": "1999",
         "publisher": "John Wiley & Sons", "city": "New York, NY, USA"},
    16: {"type": "Book", "authors": ["M. G. Morgan", "M. Henrion"],
         "title": "Uncertainty: A Guide to Dealing with Uncertainty in Quantitative Risk and Policy Analysis", "year": "1990",
         "publisher": "Cambridge University Press", "city": "Cambridge, UK"},
    17: {"type": "Book", "authors": ["G. Shafer"], "title": "A Mathematical Theory of Evidence", "year": "1976",
         "publisher": "Princeton University Press", "city": "Princeton, NJ, USA"},
    18: {"type": "ConferenceProceedings", "authors": ["M. T. Ribeiro", "S. Singh", "C. Guestrin"],
         "title": "\"Why should I trust you?\" Explaining the predictions of any classifier",
         "conference": "Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining", "year": "2016", "pages": "1135-1144",
         "city": "San Francisco, CA, USA", "publisher": "ACM"},
    19: {"type": "Report", "authors": ["J. H. Scott", "R. E. Burgan"],
         "title": "Standard fire behavior fuel models: A comprehensive set for use with Rothermel's surface fire spread model",
         "year": "2005", "institution": "USDA Forest Service, Rocky Mountain Research Station", "city": "Fort Collins, CO, USA",
         "reporttype": "Gen. Tech. Rep. RMRS-GTR-153"},
    24: {"type": "BookSection", "authors": ["D. L. Martell"], "title": "Forest fire management",
         "book": "Handbook of Operations Research in Natural Resources", "editors": ["A. Weintraub", "C. Romero", "T. Bjørndal", "R. Epstein"],
         "year": "2007", "pages": "489-509", "publisher": "Springer", "city": "New York, NY, USA"},
    25: {"type": "Book", "authors": ["R. L. Keeney", "H. Raiffa"], "title": "Decisions with Multiple Objectives: Preferences and Value Trade-Offs",
         "year": "1976", "publisher": "Wiley", "city": "New York, NY, USA"},
    26: {"type": "Book", "authors": ["S. French", "J. Maule", "N. Papamichail"], "title": "Decision Behaviour, Analysis and Support",
         "year": "2019", "publisher": "Cambridge University Press", "city": "Cambridge, UK"},
    28: {"type": "BookSection", "authors": ["A. Colorni", "A. Tsoukias"], "title": "What is a decision problem? Designing alternatives",
         "book": "Preference Disaggregation in Multiple Criteria Decision Analysis", "editors": ["N. Matsatsinis", "E. Grigoroudis"],
         "year": "2018", "pages": "1-15", "publisher": "Springer", "city": "Cham, Switzerland"},
    32: {"type": "Book", "authors": ["G. A. Klein"], "title": "Sources of Power: How People Make Decisions", "year": "2017",
         "publisher": "MIT Press", "city": "Cambridge, MA, USA"},
    35: {"type": "Misc", "authors": ["M. S. Levin"], "title": "Towards decision support technology platform for modular systems",
         "pubtitle": "arXiv:1408.5492", "year": "2014"},
    36: {"type": "ConferenceProceedings", "authors": ["P. Struss"],
         "title": "Model-based decision support systems: Conceptualization and general architecture",
         "conference": "Proc. 32nd Int. Workshop on Qualitative Reasoning (QR)", "year": "2019", "pages": ""},
    38: {"type": "ConferenceProceedings", "authors": ["E. Gizzi", "H. Amel", "W. W. Lin", "K. Rhea", "J. Sinapov"],
         "title": "Toward creative problem solving agents: Action discovery through behavior babbling",
         "conference": "Proc. IEEE Int. Conf. Development and Learning (ICDL)", "year": "2021", "pages": "1-7"},
    41: {"type": "Book", "authors": ["R. S. Sutton", "A. G. Barto"], "title": "Reinforcement Learning: An Introduction", "year": "2018",
         "publisher": "MIT Press", "city": "Cambridge, MA, USA"},
    42: {"type": "Misc", "authors": ["Y. Xie", "B. Jiang", "T. Mallick", "J. D. Bergerson", "J. K. Hutchison", "D. R. Verner"],
         "title": "WildfireGPT: Tailored large language model for wildfire analysis", "pubtitle": "arXiv:2402.07877", "year": "2024"},
    44: {"type": "JournalArticle", "authors": ["P. Dubey", "P. Dubey"],
         "title": "Bridging spatiotemporal wildfire prediction and decision modeling using transformer networks and fuzzy inference systems",
         "journal": "", "year": "2025", "volume": "15", "pages": "103498"},
    49: {"type": "Report", "authors": ["Y. Tan"],
         "title": "Trustworthy intelligence fusion for search and rescue: Designing grounded and secure multi-agent AI for mission decision support",
         "year": "2025", "institution": "California Polytechnic State Univ.", "city": "San Luis Obispo, CA, USA", "reporttype": "M.S. thesis"},
}


def load():
    recs = {}
    for line in open(TXT, encoding="utf-8"):
        n, s = line.rstrip("\n").split("\t", 1); n = int(n)
        recs[n] = OVERRIDE.get(n) or parse_journal(s)
    return recs


if __name__ == "__main__":
    for n, r in sorted(load().items()):
        print(n, r["type"], r["authors"][0], "|", r.get("journal") or r.get("publisher") or r.get("conference") or r.get("institution") or r.get("pubtitle"), "|", r.get("volume"), r.get("issue"), r.get("pages"), r["year"])
