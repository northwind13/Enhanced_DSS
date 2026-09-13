import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=pd.read_csv('experiments/out/ladder_runs.csv')
d['popp']=d.pop_affected*9e-4
arms=['Test0','F5','F5Ev','F5EvAI']; labels=[r'$T_0$',r'$T_{F5}$',r'$T_{F5+Ev}$',r'$T_{DisasterAware}$']
sc=d.groupby(['arm','scenario']).mean(numeric_only=True)   # scenario means
m=sc.groupby('arm').mean().loc[arms]
W=np.array([1,1,1,0.2,0.2])/3.4
RED,GRN,BLU,PUR,GRY='#c0392b','#27ae60','#2980b9','#8e44ad','#95a5a6'
def bars(ax, series, colors, names, fmt, ylim, ylabel, title, share=True, annot_top=None):
    n=len(series); w=0.8/n; x=np.arange(len(arms))
    for k,(s,c,nm) in enumerate(zip(series,colors,names)):
        v=np.array(s); h=100*v/v[0] if share else v
        b=ax.bar(x+(k-(n-1)/2)*w, h, w, color=c, label=nm)
        for xi,hi,vi in zip(x+(k-(n-1)/2)*w,h,v):
            ax.text(xi, hi+ (1.5 if share else 0.003), fmt(vi), rotation=90, ha='center', va='bottom', fontsize=8)
    if share: ax.axhline(100, color='0.3', ls='--', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10); ax.set_ylim(*ylim); ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=10); ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.14), ncol=n, frameon=False, fontsize=8, handlelength=1.5, columnspacing=1.2)
# Fig 5.16 outcome
fig,ax=plt.subplots(figsize=(6.06,4.42),dpi=200)
bars(ax,[m.burned_ha,m.forest_ha,m.popp],[RED,GRN,BLU],['Burned (ha)','Burned Forest (ha)','AffectedPop.'],lambda v:f'{v:,.0f}' if v>1000 else f'{v:.1f}',(0,128),'share of the no-DSS run $T_0$ (%)','S1–S5')
plt.setp(ax.get_xticklabels(), rotation=15); fig.tight_layout(); fig.savefig('fig5_16_outcome_average.png'); plt.close()
# Fig 5.17 physical
fig,ax=plt.subplots(figsize=(6.07,3.67),dpi=200)
bars(ax,[m.end_j_burn,m.end_j_asset,m.end_j_pop],[RED,GRN,BLU],['burned area','asset loss','population'],lambda v:f'{v:.3f}',(0,128),'physical decision cost, share of $T_0$ (%)','Avg')
fig.tight_layout(); fig.savefig('fig5_17_Jphys_average.png'); plt.close()
# Fig 5.18 total weighted
fig,ax=plt.subplots(figsize=(6.07,3.67),dpi=200)
terms=[m.end_j_burn*W[0],m.end_j_asset*W[1],m.end_j_pop*W[2],m.end_j_resp*W[3],m.end_j_delay*W[4]]
bars(ax,terms,[RED,GRN,BLU,PUR,GRY],['burned area','asset loss','population','response','delay'],lambda v:f'{v:.3f}',(0,0.33),'weighted cost term (sums to $J_{total}$)','Avg',share=False)
for i,a in enumerate(arms): ax.text(i,0.30,f'$J_{{total}}$ = {m.end_j_total[a]:.3f}',ha='center',fontsize=7)
fig.tight_layout(); fig.savefig('fig5_18_Jtotal_average.png'); plt.close()
print(m[['burned_ha','forest_ha','popp','end_j_burn','end_j_asset','end_j_pop','end_j_resp','end_j_delay','end_j_total','end_j_phys']].round(3).to_string())
print('check jtotal', (sum(terms)).round(3).tolist())
