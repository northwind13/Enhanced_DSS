import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig,ax=plt.subplots(figsize=(6.5,4.0),dpi=200); ax.set_xlim(0,10); ax.set_ylim(0,6.2); ax.axis('off')
def box(x,y,w,h,fc,ec,lines,weights=None,sizes=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.15",fc=fc,ec=ec,lw=1.2))
    n=len(lines)
    for i,l in enumerate(lines):
        ax.text(x+w/2,y+h-(i+0.5)*h/n,l,ha='center',va='center',fontsize=(sizes[i] if sizes else 8.8),fontweight=(weights[i] if weights else 'normal'))
box(0.2,4.9,3.3,1.05,'#dae8fc','#6c8ebf',['asset exposure risk','(level 2 concept)'])
box(6.5,4.9,3.3,1.05,'#dae8fc','#6c8ebf',['evacuation pressure','(level 3 concept)'])
box(0.2,2.65,9.6,1.35,'#d5e8d4','#82b366',['interface_exposure (generated, level 3 intermediate concept)','0.50 · asset exposure risk + 0.50 · evacuation pressure','normalized weighted sum, confidence-gated'],weights=['bold','normal','normal'])
box(0.2,0.1,9.6,1.7,'#ffffff','#82b366',['Rule G14 (generated)','IF interface_exposure VH ∧ suppression_feasibility VH','THEN asset_protection 0.9; containment_line 0.8;','resource_deployment 0.7'],weights=['bold','normal','normal','normal'])
for x0,x1 in [(1.85,3.6),(8.15,6.4)]:
    ax.add_patch(FancyArrowPatch((x0,4.9),(x1,4.02),arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#333'))
    ax.text((x0+x1)/2+(0.25 if x0<x1 else -0.25),4.5,'0.5',color='#c55a11',fontsize=9.5,ha='center')
ax.add_patch(FancyArrowPatch((5,2.65),(5,1.82),arrowstyle='-|>',mutation_scale=12,lw=1.2,color='#333'))
fig.savefig('fig5_20_concept_rule_v3.png',bbox_inches='tight',facecolor='white'); print('ok')
