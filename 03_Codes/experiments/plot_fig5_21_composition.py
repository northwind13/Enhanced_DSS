import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
rows=[("downwind_containment_shield",{"containment line":1.0,"suppression effort":0.7}),
("downwind_retardant_shield",{"retardant drop":1.0,"containment line":0.8}),
("drafting_sustained_attack",{"water drafting":1.0,"suppression effort":0.9}),
("head_knockdown",{"water drafting":1.0,"suppression effort":0.9,"retardant drop":0.6}),
("wet_containment_line",{"water drafting":1.0,"containment line":0.8}),
("counterfire_strip",{"tactical burn":1.0,"containment line":0.6}),
("drafting_retardant_shuttle",{"retardant drop":1.0,"water drafting":0.6})]
cols=["containment line","suppression effort","retardant drop","water drafting","tactical burn"]
M=np.zeros((len(rows),len(cols)))
for i,(n,d) in enumerate(rows):
    for k,v in d.items(): M[i,cols.index(k)]=v
fig,ax=plt.subplots(figsize=(10.1,5.05))
im=ax.imshow(M,cmap="Blues",vmin=0,vmax=1,aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols,rotation=30,ha="right",fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows],fontsize=10)
for i in range(len(rows)):
    for j in range(len(cols)):
        if M[i,j]>0: ax.text(j,i,f"{M[i,j]:.1f}",ha="center",va="center",color="white" if M[i,j]>0.5 else "#222",fontsize=10)
ax.set_xticks(np.arange(-.5,len(cols),1),minor=True); ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
ax.grid(which="minor",color="white",lw=2); ax.tick_params(which="minor",length=0)
ax.set_title("composition weights of the generated macro interventions",fontsize=11)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.03); cb.set_label("weight on the base intervention",fontsize=10)
fig.text(0.5,0.01,"generated concept (level 4): interface exposure = 0.50 asset exposure risk + 0.50 evacuation pressure",ha="center",fontsize=10,style="italic")
fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig("fig5_21_composition_v3.png",dpi=200,facecolor="white"); print("ok")
