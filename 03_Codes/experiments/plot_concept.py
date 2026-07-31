"""Generated intermediate concept figure for Section 5.4.3.2.
interface_exposure (Layer 4) = 0.5 asset_exposure_risk + 0.5 evacuation_pressure,
cited by the generated rule G13. Reads the definition from the ledger."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
FIG=os.path.join(ROOT,"..","01_Thesis","figures","fig_concept_interface_exposure.png")
st=json.load(open(os.path.join(ROOT,"logs","dss_generated_state.json")))
c=st["genai_concepts"][0]
inputs=c["inputs"]  # name, weight

fig,ax=plt.subplots(figsize=(8.2,4.6)); ax.axis("off")
ax.set_xlim(0,10); ax.set_ylim(0,10)
def box(x,y,w,h,text,fc,ec,fs=10,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
                fc=fc,ec=ec,lw=1.5))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,
            fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2,label=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,
                lw=1.4,color="#555"))
    if label: ax.text((x1+x2)/2+0.25,(y1+y2)/2,label,fontsize=9,color="#b0530a")
# Layer 3 input concepts
box(0.6,7.9,3.6,1.4, inputs[0]["name"].replace("_"," ")+"\n(Layer 3 concept)","#dae8fc","#6c8ebf",10)
box(5.8,7.9,3.6,1.4, inputs[1]["name"].replace("_"," ")+"\n(Layer 3 concept)","#dae8fc","#6c8ebf",10)
# Layer 4 concept
box(2.6,4.7,4.8,1.8,
    c["name"].replace("_"," ")+"  (Layer 4)\n"+" + ".join(f"{i['weight']:.2f} · {i['name'].replace('_',' ')}" for i in inputs)
    +"\nnormalized weighted sum, confidence-gated","#d5e8d4","#82b366",10,bold=True)
# rule
box(0.6,1.2,8.8,1.9,
    "Rule G13 (generated):\nIF interface_exposure VH  ∧  suppression_feasibility VH\n"
    "THEN asset_protection 0.9; containment_line 0.8; resource_deployment 0.7",
    "#ffffff","#82b366",9.5)
arrow(2.4,7.9,4.2,6.5,"0.5"); arrow(7.6,7.9,5.8,6.5,"0.5")
arrow(5.0,4.7,5.0,3.1)
ax.text(5.0,9.7,"Generated intermediate concept and the rule that cites it",
        ha="center",fontsize=11,fontweight="bold")
fig.tight_layout()
os.makedirs(os.path.dirname(FIG),exist_ok=True)
fig.savefig(FIG,dpi=220,bbox_inches="tight"); print("written",FIG)
