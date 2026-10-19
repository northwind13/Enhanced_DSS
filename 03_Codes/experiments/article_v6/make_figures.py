"""Build every article figure (block diagrams and result charts) as PNG for the manuscript and as ONE editable draw.io file,
one page per figure in article order."""
import figs, resfigs_native
from diag import drawio_file
order = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "Fig7", "Fig8", "Fig9", "Fig10", "Fig11", "Fig12", "Fig13"]
allp = figs.pages + resfigs_native.PAGES
pages = [next(p for p in allp if p[0].startswith(k + " ")) for k in order]
drawio_file(pages, "/home/claude/v6/fig/DisasterAware_Article_Figures.drawio")
print("pages:", [p[0] for p in pages])
