"""v22_v1 -> v22_v2: drop City, Publisher, DOI and URL from every source; rebuild the rendered bibliography table."""
import sys, re
from lxml import etree
sys.argv = [sys.argv[0], sys.argv[1]]
D = sys.argv[1]
B = "http://schemas.openxmlformats.org/officeDocument/2006/bibliography"; NS = {"b": B}
src = etree.parse(f"{D}/customXml/item2.xml"); root = src.getroot()
for s in root.findall("b:Source", NS):
    tag = s.find("b:Tag", NS).text
    for k in ("City", "Publisher", "DOI", "URL", "StateProvince", "CountryRegion", "Issue"):
        for e in s.findall(f"b:{k}", NS):
            if tag == "Rothermel72" and k == "City": e.text = "Res. Pap. INT-115"; continue
            s.remove(e)
src.write(f"{D}/customXml/item2.xml", xml_declaration=True, encoding="UTF-8", standalone=True)
# re-render with the shorten script's renderer (no further field edits)
import importlib.util
spec = importlib.util.spec_from_file_location("sh", "shorten_refs_v22.py"); code = open("shorten_refs_v22.py").read()
code = code.split("# render like Word's IEEE 2006 style")[1]
code = code.replace('out.append((f"{a}, ", False)); out.append((t + ", ", True)); out.append((f\'{f(s, "City")}: {f(s, "Publisher")}, {y}. \', False))',
                    'out.append((f"{a}, ", False)); out.append((t + ", ", True)); out.append((f"{y}. ", False))')
pre = open("shorten_refs_v22.py").read().split("# tag -> field updates")[0]
exec(pre + "\nroot = etree.parse(f'{D}/customXml/item2.xml').getroot()\n" + code)
