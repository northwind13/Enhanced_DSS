import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
INK="#33322e"; MUTED="#77746c"
fig,(a,b)=plt.subplots(1,2,figsize=(14,6),gridspec_kw=dict(width_ratios=[1,1.3]))
# left: evolving fuzzy stages
tried=[2304,2272]; acc=[1480,552]
x=np.arange(2); w=0.36
a.bar(x-w/2,tried,w,color="#c9d6ea",edgecolor="#2f5f9e",label="tried")
a.bar(x+w/2,acc,w,color="#2f6fc0",edgecolor="#2f5f9e",label="accepted")
for i in range(2):
    a.text(x[i]-w/2,tried[i]+30,f"{tried[i]:,}".replace(","," "),ha="center",fontsize=10,color=INK)
    a.text(x[i]+w/2,acc[i]+30,f"{acc[i]:,} ({100*acc[i]/tried[i]:.0f}%)".replace(","," "),ha="center",fontsize=10,color=INK)
a.set_xticks(x); a.set_xticklabels(["stage ①\nconsequent step","stage ②\nresolution increase"],fontsize=10)
a.set_ylabel("adaptation trials",fontsize=10); a.set_ylim(0,2750)
a.set_title("evolving fuzzy stages",fontsize=11,color=INK)
a.legend(frameon=False,fontsize=10,loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=2)
a.grid(axis="y",color="#e3e1dc",lw=0.8); a.set_axisbelow(True)
for s in ("top","right"): a.spines[s].set_visible(False)
# right: gate cascade
labels=["proposals","rejected at G2\n(duplicate cell / package name)","rejected at G3 to G5\n(forecast, robustness, margin)","admitted\n(272 rules + 120 packages)"]
vals=[1144,222,530,392]
cols=["#c9d6ea","#d0453a","#d0453a","#2e9e4a"]
y=np.arange(4)[::-1]
b.barh(y,vals,color=cols,edgecolor="none",height=0.62)
for yi,v in zip(y,vals): b.text(v+15,yi,f"{v:,}".replace(","," "),va="center",fontsize=10,color=INK)
b.set_yticks(y); b.set_yticklabels(labels,fontsize=10)
b.set_xlabel("proposals",fontsize=10); b.set_xlim(0,1300)
b.set_title("generative stage ③ gate cascade",fontsize=11,color=INK)
b.grid(axis="x",color="#e3e1dc",lw=0.8); b.set_axisbelow(True)
for s in ("top","right"): b.spines[s].set_visible(False)
from matplotlib.patches import Patch
b.legend(handles=[Patch(color="#c9d6ea",label="proposals made"),Patch(color="#d0453a",label="rejected by a gate"),Patch(color="#2e9e4a",label="admitted")],
         frameon=False,fontsize=10,loc="upper center",bbox_to_anchor=(0.5,-0.14),ncol=3,title="colour key",title_fontsize=9)
fig.tight_layout(); fig.savefig("fig5_19_funnel_v3.png",dpi=200,facecolor="white"); print("ok")
