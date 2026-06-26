'use strict';
const $=id=>document.getElementById(id);
const api=(u,b)=>fetch(u,{method:b?'POST':'GET',headers:{'Content-Type':'application/json'},body:b?JSON.stringify(b):undefined}).then(r=>r.json());

let WORLD={W:100,H:100,fuel:[],value:[],slope:[],access:[],cat:'',legend:[],type_legend:[],stations:[],grid_n:2,n_responders:6,capacity:150,cell_m:30,dt_min:2};
let CATCOL={},CATICON={},TYPECOL={m1:'#22b8cf',m2:'#4a8fd0',m3:'#d65cc8'};
let SNAP=null,viewIndex=0,count=1,timer=null,mode=null,bg=document.createElement('canvas');
let conceptCache={};

function curParams(){return{
  seed:+$('seed').value,H:+$('grid').value,W:+$('grid').value,forest_frac:+$('forest').value,
  fuel_density:+$('fuelDensity').value,wind_speed:+$('windSpeed').value,wind_dir:+$('windDir').value,
  humidity:+$('humidity').value,spotting:+$('spotting').value,n_responders:+$('responders').value,
  eta:+$('eta').value,eps:+$('eps').value,regions:+$('regions').value,
  event_trigger:+$('eventTrigger').value,dss_enabled:$('dssEnabled').checked};}
const layerSel=()=>document.querySelector('input[name=layer]:checked').value;
const clickMode=()=>document.querySelector('input[name=clickmode]:checked').value;
const zoomPx=()=>+$('zoom').value;
const conceptSel=()=>$('conceptLayer').value;

function gradColor(layer,v){const t=v/255;
  if(layer==='fuel')return `rgb(28,${36+t*150|0},44)`;
  if(layer==='value')return `rgb(${50+t*180|0},42,42)`;
  if(layer==='slope')return `rgb(${30+t*120|0},${30+t*120|0},${36+t*120|0})`;
  return '#0b0f14';}
function heat(v){const t=v/255; return `rgba(${40+t*215|0},${(1-t)*180+30|0},40,${0.15+t*0.6})`;}

function buildBackground(){
  const W=WORLD.W,H=WORLD.H,px=zoomPx(),layer=layerSel();
  bg.width=W*px; bg.height=H*px; const ctx=bg.getContext('2d');
  for(let i=0;i<W*H;i++){let col;
    if(layer==='landuse')col=CATCOL[WORLD.cat.charCodeAt(i)-48]||'#0b0f14';
    else col=gradColor(layer,(WORLD[layer]||[])[i]||0);
    ctx.fillStyle=col; ctx.fillRect((i%W)*px,(i/W|0)*px,px+0.6,px+0.6);}
  if($('showIcons').checked && px>=8){
    ctx.font=`${Math.round(px*0.82)}px "Segoe UI Emoji",sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle';
    const STR={5:1,6:1,7:1},VEG={2:1,3:1,4:1,8:1},st=px>=14?2:3;
    for(let i=0;i<W*H;i++){const c=WORLD.cat.charCodeAt(i)-48;const ic=CATICON[c];if(!ic)continue;
      const x=i%W,y=i/W|0; if(STR[c]||(VEG[c]&&x%st===0&&y%st===0))ctx.fillText(ic,(x+0.5)*px,(y+0.58)*px);}}
  if($('showRegions').checked&&WORLD.grid_n>1){ctx.strokeStyle='rgba(255,255,255,0.35)';ctx.lineWidth=1;
    ctx.font=`${Math.max(9,px)}px sans-serif`;ctx.fillStyle='rgba(255,255,255,0.6)';ctx.textAlign='left';ctx.textBaseline='top';
    const g=WORLD.grid_n;for(let r=0;r<g;r++)for(let cI=0;cI<g;cI++){
      const x0=Math.floor(cI*W/g)*px,y0=Math.floor(r*H/g)*px;
      ctx.strokeRect(x0,y0,Math.floor((cI+1)*W/g)*px-x0,Math.floor((r+1)*H/g)*px-y0);ctx.fillText('R'+(r*g+cI+1),x0+3,y0+2);}}
  if($('showStations').checked&&px>=4){ctx.font=`${Math.max(11,Math.round(px*1.3))}px "Segoe UI Emoji",sans-serif`;
    ctx.textAlign='center';ctx.textBaseline='middle';
    for(const s of WORLD.stations){ctx.fillStyle='rgba(0,0,0,0.35)';ctx.beginPath();ctx.arc((s.x+0.5)*px,(s.y+0.5)*px,px*0.9,0,7);ctx.fill();
      ctx.fillText('🚒',(s.x+0.5)*px,(s.y+0.6)*px);}}
}

function drawMap(cv,m,isManaged){
  const W=WORLD.W,H=WORLD.H,px=zoomPx(); cv.width=W*px; cv.height=H*px;
  const ctx=cv.getContext('2d'); ctx.drawImage(bg,0,0);
  // concept heatmap overlay (managed, live)
  if(isManaged){const cn=conceptSel(); const cd=conceptCache[cn];
    if(cn&&cd&&cd.length===W*H){for(let i=0;i<W*H;i++){const v=cd[i];if(v>20){ctx.fillStyle=heat(v);ctx.fillRect((i%W)*px,(i/W|0)*px,px+0.6,px+0.6);}}}}
  if(isManaged&&SNAP&&SNAP.dss&&SNAP.dss.enabled&&!conceptSel()){
    const showAct=$('showActions').checked;
    for(let i=0;i<W*H;i++){
      if(showAct){const d=m.dom.charCodeAt(i)-48;if(d>0){ctx.fillStyle=[null,TYPECOL.m1,TYPECOL.m2,TYPECOL.m3][d]+'c0';ctx.fillRect((i%W)*px,(i/W|0)*px,px+0.6,px+0.6);}}
      else{const s=m.supp.charCodeAt(i)-48;if(s>0){ctx.fillStyle=`rgba(34,184,207,${0.12+s/9*0.6})`;ctx.fillRect((i%W)*px,(i/W|0)*px,px+0.6,px+0.6);}}}}
  for(let i=0;i<W*H;i++){const c=m.codes.charCodeAt(i)-48;
    if(c===1){ctx.fillStyle='rgba(35,20,14,0.85)';ctx.fillRect((i%W)*px,(i/W|0)*px,px+0.6,px+0.6);}
    else if(c===2){ctx.fillStyle='#ff5a1f';ctx.fillRect((i%W)*px,(i/W|0)*px,px+0.6,px+0.6);}}
}
function chart(cv,base,dss,max){const ctx=cv.getContext('2d'),W=cv.width,H=cv.height,pad=18;ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#2a3340';ctx.beginPath();ctx.moveTo(pad,H-pad);ctx.lineTo(W-4,H-pad);ctx.stroke();
  const n=base.length;if(n<2)return;const mx=Math.max(max,...base,...dss,1);
  const X=i=>pad+i/(n-1)*(W-pad-6),Y=v=>(H-pad)-v/mx*(H-pad-8);
  const ln=(a,c)=>{ctx.strokeStyle=c;ctx.lineWidth=2;ctx.beginPath();a.forEach((v,i)=>i?ctx.lineTo(X(i),Y(v)):ctx.moveTo(X(i),Y(v)));ctx.stroke();};
  ln(base,'#b0392b');ln(dss,'#2fae6e');const vi=Math.min(viewIndex,n-1);ctx.fillStyle='#f0883e';ctx.beginPath();ctx.arc(X(vi),Y(dss[vi]),3,0,7);ctx.fill();
  ctx.fillStyle='#8b97a7';ctx.font='10px sans-serif';ctx.fillText(mx.toFixed(0),2,Y(mx)+8);}

const CATSHOW=[['forest','Forest'],['grove','Grove'],['agri','Agri'],['animal','Livestock'],['resid','Residential'],['urban','City'],['crit','Critical']];
function burncats(el,by){el.innerHTML=CATSHOW.map(([k,n])=>{const v=(by[k]||{}).burned||0;return v?`<span class="bc">${n} <b>${v}</b></span>`:'';}).join('');}

async function ensureConcept(){const cn=conceptSel();
  if(!cn||!SNAP||!SNAP.live||!SNAP.dss||!SNAP.dss.enabled){return;}
  if(conceptCache.__step===SNAP.step&&conceptCache[cn]){return;}
  const r=await api('/api/concept',{name:cn}); conceptCache={__step:SNAP.step,[cn]:r.data};}

async function render(snap){
  if(!snap)return; SNAP=snap; viewIndex=snap.index; count=snap.count;
  $('stepNum').textContent=snap.step; $('stepMax').textContent=count-1; $('scrub').max=count-1; $('scrub').value=snap.index;
  $('clock').textContent=snap.minute+' min'; $('liveTag').classList.toggle('hidden',!snap.live);
  $('ckptTag').classList.toggle('hidden',snap.checkpoint==null);
  await ensureConcept();
  drawMap($('cvBase'),snap.baseline,false); drawMap($('cvDss'),snap.managed,true);
  const b=snap.baseline,m=snap.managed,d=snap.dss;
  $('bBurned').textContent=b.burned+'%'; $('bHa').textContent=b.ha+' ha'; $('bLoss').textContent=b.loss; $('bActive').textContent=b.active;
  $('dBurned').textContent=m.burned+'%'; $('dHa').textContent=m.ha+' ha'; $('dLoss').textContent=m.loss;
  $('dQ').textContent=d.Q==null?(d.idle?'idle':'–'):d.Q;
  burncats($('bCats'),b.by_cat); burncats($('dCats'),m.by_cat);
  $('frCount').textContent=WORLD.n_responders; $('capTot').textContent=WORLD.capacity; $('capUsed').textContent=d.used==null?0:d.used;
  const sv=snap.savings; $('savings').innerHTML=snap.step===0?'':
    `<span class="sv ok">value protected <b>${sv.value_protected}</b></span><span class="sv ok">area protected <b>${sv.area_protected_ha} ha</b></span><span class="sv ok">loss cut <b>${sv.pct_loss_cut}%</b></span>`;
  const eta=+$('eta').value,as=$('acceptStatus');
  if(d.idle){as.innerHTML='DSS <b>idle</b> — waiting for trigger';as.className='rb';}
  else if(d.Q==null){as.textContent='—';as.className='rb';}
  else if(d.accepted){as.innerHTML=`Plan <b>ACCEPTED</b> — Q ${d.Q} ≥ η ${eta.toFixed(2)}`;as.className='rb ok';}
  else{as.innerHTML=`Plan <b>ATTENUATED</b> (fail-safe) — Q ${d.Q} &lt; η ${eta.toFixed(2)}`;as.className='rb warn';}
  $('typeLegend').innerHTML=WORLD.type_legend.map(t=>`<span class="tl"><i class="sw" style="background:${t.color}"></i>${t.label}</span>`).join('');
  if(d.summary){const parts=['m1','m2','m3'].map(t=>{const s=d.summary[t];const cats=Object.entries(s.by_cat).map(([k,v])=>`${k} ${v}`).join(', ');
      return `<b style="color:${s.color}">${s.label}</b>: ${s.cells} cells${cats?' ('+cats+')':''}`;});
    $('decisionText').innerHTML=`<b>k=${snap.step} (${snap.minute} min).</b> `+parts.join('. ')+'.';
    $('qbars').innerHTML=Object.entries(d.q).map(([k,v])=>`<span class="q">${k} <b>${v}</b></span>`).join('');}
  else if(d.idle){$('decisionText').innerHTML='DSS is in event-triggered mode and idle until the fire reaches the trigger size.';$('qbars').innerHTML='';}
  else if(d.enabled===false){$('decisionText').innerHTML='DSS is off; the right panel mirrors the baseline.';$('qbars').innerHTML='';}
  else{$('decisionText').textContent='Start a fire to see the decision support system act.';$('qbars').innerHTML='';}
  const s=snap.series; chart($('chBurned'),s.base_burned,s.dss_burned,100); chart($('chLoss'),s.base_loss,s.dss_loss,snap.asset_total);
}

async function doReset(){stop();conceptCache={};
  const r=await api('/api/reset',curParams()); WORLD=r.layers;
  CATCOL={};CATICON={}; WORLD.legend.forEach(c=>{CATCOL[c.id]=c.color;CATICON[c.id]=c.icon;});
  WORLD.type_legend.forEach(t=>TYPECOL[t.key]=t.color); $('gridVal').textContent=`${WORLD.W}×${WORLD.H}`;
  buildBackground(); render(r.view); $('connStatus').textContent='ready'; $('connStatus').classList.add('ok');}
const pushParams=()=>api('/api/params',curParams());
async function goStep(n){conceptCache={};render(await api('/api/step',{n}));}
async function goto(i){render(await api('/api/goto',{i}));}
async function clearFire(){stop();conceptCache={};render(await api('/api/clear_fire',{}));}
function igniteAt(x,y){conceptCache={};api('/api/ignite',{x,y,radius:+$('ignRadius').value}).then(render);}
async function inspectAt(x,y){const r=await api('/api/inspect',{x,y});showInspect(r);}
function showInspect(r){
  if(!r){$('inspBody').textContent='no data';return;}
  let h=`<div class="ihead">cell (${r.x},${r.y}) — <b>${r.landuse}</b>${r.burning?' 🔥 burning':''}</div>`;
  h+=`<div class="irow">fuel ${r.fuel} · value ${r.value}${r.reach!=null?' · reach '+r.reach:''}</div>`;
  if(r.features){h+='<div class="isec">Features</div>'+r.features.map(([n,v])=>`<span class="ip">${n} <b>${v}</b></span>`).join('');
    h+='<div class="isec">Concepts</div>'+r.concepts.map(([n,v])=>`<span class="ip">${n} <b>${v}</b></span>`).join('');
    h+=`<div class="isec">Action chosen: <b>${r.action}</b> (effort ${r.applied})</div>`;
    for(const t of ['m1','m2','m3']){const ty=r.types[t];const fr=ty.fired.map(f=>`R${f.rule} w${f.weight}→${f.level}`).join(', ')||'no rule fired';
      h+=`<div class="irow"><b style="color:${ty.color}">${ty.label}</b> degree ${ty.degree} <span class="muted">[${fr}]</span></div>`;}}
  else h+=`<div class="irow muted">${r.note||''}</div>`;
  $('inspBody').innerHTML=h;}

function tick(){if(mode==='rew'){viewIndex>0?goto(viewIndex-1):stop();return;}
  const mult=mode==='ff'?3:1; if(viewIndex<count-1)goto(Math.min(count-1,viewIndex+mult));else goStep(mult);
  if(SNAP&&SNAP.live&&SNAP.baseline.active===0&&SNAP.managed.active===0&&SNAP.step>3)stop();}
function start(m){stop();mode=m;const sp=+$('speed').value;timer=setInterval(tick,m==='rew'?Math.max(60,sp/2):sp);
  if(m==='play'){$('btnPlay').classList.add('playing');$('btnPlay').innerHTML='⏸ Pause';}}
function stop(){if(timer)clearInterval(timer);timer=null;mode=null;$('btnPlay').classList.remove('playing');$('btnPlay').innerHTML='▶ Play';}
const playToggle=()=>timer&&mode==='play'?stop():start('play');

function windArrow(){const t=(+$('windDir').value)*Math.PI/180,dx=Math.sin(t),dy=Math.cos(t),ang=Math.atan2(dy,dx)*180/Math.PI;
  $('windArrow').style.transform=`rotate(${ang}deg)`;const dirs=['E','SE','S','SW','W','NW','N','NE'];$('windHead').textContent='fire heads '+dirs[((Math.round(ang/45)%8)+8)%8];}
function fitView(){const wrap=$('cvBase').parentElement;const px=Math.max(2,Math.min(24,Math.floor((wrap.clientWidth-4)/WORLD.W)));$('zoom').value=px;$('zoomVal').textContent=px+'px';buildBackground();render(SNAP);}

function dl(url,name){const a=document.createElement('a');a.href=url;a.download=name||'';document.body.appendChild(a);a.click();a.remove();}

function bindCanvas(cv){
  let down=null;
  cv.addEventListener('mousedown',e=>{down={x:e.clientX,y:e.clientY,sl:cv.parentElement.scrollLeft,st:cv.parentElement.scrollTop};});
  cv.addEventListener('mousemove',e=>{if(!down)return;const dx=e.clientX-down.x,dy=e.clientY-down.y;
    if(Math.abs(dx)+Math.abs(dy)>4){cv.parentElement.scrollLeft=down.sl-dx;cv.parentElement.scrollTop=down.st-dy;down.moved=true;}});
  window.addEventListener('mouseup',e=>{if(!down){return;}
    if(!down.moved){const r=cv.getBoundingClientRect();const x=Math.floor((e.clientX-r.left)/r.width*WORLD.W),y=Math.floor((e.clientY-r.top)/r.height*WORLD.H);
      if(x>=0&&x<WORLD.W&&y>=0&&y<WORLD.H){clickMode()==='inspect'?inspectAt(x,y):igniteAt(x,y);}}
    down=null;});
}

const HELP=`<p>DisasterAware shows a wildfire with <b>no intervention</b> (left) vs the <b>concept-based DSS</b> (right).
Each cell is ${30} m; the clock is in minutes.</p>
<h3>Quick start</h3><p>Pick a <b>preset</b> or click a map to ignite, press <code>Play</code>. Use the timeline to rewind/scrub.
Keys: <code>Space</code> play, <code>←/→</code> step.</p>
<h3>Explain a decision</h3><p>Set <b>Click = inspect</b> and click any cell to see its six features, four concepts, the fuzzy rules that fired,
and the action the DSS chose. The <b>Concept overlay</b> shows how the DSS "sees" threat / feasibility / exposure / urgency.</p>
<h3>Intervention types (M=3)</h3><ul>
<li><b style="color:#22b8cf">Direct suppression</b> (α=1.0) — put out burning cells.</li>
<li><b style="color:#4a8fd0">Preventive fuel reduction</b> (α=0.7) — fuel break ahead of the front.</li>
<li><b style="color:#d65cc8">Asset protection</b> (α=0.9) — defend homes, city, critical facilities.</li></ul>
<h3>Acceptance threshold η</h3><p>The DSS scores its plan as <code>Q∈[0,1]</code> (spread 0.35, asset 0.30, resource 0.20, timeliness 0.15).
If <code>Q≥η</code> it is <b>ACCEPTED</b>; else a fail-safe makes it <b>ATTENUATED</b>. Status is shown under the maps.</p>
<h3>Realism</h3><p>Wind drives a Rothermel-type spread; <b>humidity</b> dampens it; <b>spotting</b> throws embers across firebreaks.
<b>First responders</b> (🚒) dispatch suppression — cells far from a station are harder to defend.
<b>Event-triggered</b> activation keeps the DSS idle until the fire reaches a chosen size.</p>
<h3>Tools</h3><p>Set a <b>checkpoint</b> and <b>restore</b> to replay a what-if; export the run as <b>CSV</b>, the scenario as <b>JSON</b>,
or the current view as <b>PNG</b>. Full method: <code>DSS_DECISION_RATIONALE.md</code>.</p>`;

window.addEventListener('DOMContentLoaded',()=>{
  const bind=(id,vid,f)=>{const e=$(id);const u=()=>$(vid).textContent=f(e.value);e.addEventListener('input',u);u();};
  bind('grid','gridVal',v=>`${v}×${v}`);bind('forest','forestVal',v=>(+v).toFixed(1));bind('fuelDensity','fuelDensVal',v=>(+v).toFixed(2));
  bind('zoom','zoomVal',v=>v+'px');bind('windSpeed','windSpdVal',v=>(+v).toFixed(2));bind('windDir','windDirVal',v=>v+'°');
  bind('humidity','humVal',v=>(+v).toFixed(2));bind('spotting','spotVal',v=>(+v).toFixed(1));
  bind('responders','frVal',v=>v);bind('eta','etaVal',v=>(+v).toFixed(2));bind('eps','epsVal',v=>(+v).toFixed(2));
  windArrow();
  $('windDir').addEventListener('input',()=>{windArrow();pushParams();});
  ['windSpeed','eta','eps','humidity','spotting'].forEach(id=>$(id).addEventListener('change',pushParams));
  $('eventTrigger').addEventListener('change',pushParams); $('dssEnabled').addEventListener('change',pushParams);
  $('responders').addEventListener('change',doReset);
  ['seed','grid','forest','fuelDensity'].forEach(id=>$(id).addEventListener('change',doReset));
  $('regions').addEventListener('change',doReset);
  document.querySelectorAll('input[name=layer]').forEach(r=>r.addEventListener('change',()=>{buildBackground();render(SNAP);}));
  $('conceptLayer').addEventListener('change',()=>{conceptCache={};render(SNAP);});
  ['zoom','showIcons','showStations','showRegions'].forEach(id=>$(id).addEventListener('change',()=>{buildBackground();render(SNAP);}));
  $('showActions').addEventListener('change',()=>render(SNAP));
  $('btnReset').addEventListener('click',doReset);$('btnFit').addEventListener('click',fitView);
  $('btnPlay').addEventListener('click',playToggle);
  $('btnStep').addEventListener('click',()=>{stop();viewIndex<count-1?goto(viewIndex+1):goStep(1);});
  $('btnBack').addEventListener('click',()=>{stop();if(viewIndex>0)goto(viewIndex-1);});
  $('btnFF').addEventListener('click',()=>start('ff'));$('btnRewind').addEventListener('click',()=>start('rew'));
  $('btnFirst').addEventListener('click',()=>{stop();goto(0);});$('btnLast').addEventListener('click',()=>{stop();goto(count-1);});
  $('scrub').addEventListener('input',e=>{stop();goto(+e.target.value);});
  $('btnClear').addEventListener('click',clearFire);
  document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',async()=>{stop();conceptCache={};
    const r=await api('/api/preset',{name:b.dataset.preset});WORLD=r.layers;
    CATCOL={};CATICON={};WORLD.legend.forEach(c=>{CATCOL[c.id]=c.color;CATICON[c.id]=c.icon;});buildBackground();render(r.view);
    // reflect changed params
    $('windSpeed').value=r.params.wind_speed;$('windDir').value=r.params.wind_dir;$('humidity').value=r.params.humidity;windArrow();}));
  $('btnCkpt').addEventListener('click',async()=>{await api('/api/checkpoint',{});render(await api('/api/state'));});
  $('btnRestore').addEventListener('click',async()=>{stop();render(await api('/api/restore',{}));});
  $('btnCsv').addEventListener('click',()=>dl('/api/export_csv','disasteraware_run.csv'));
  $('btnSave').addEventListener('click',()=>dl('/api/save_scenario','disasteraware_scenario.json'));
  $('btnPng').addEventListener('click',()=>dl($('cvDss').toDataURL('image/png'),'disasteraware_view.png'));
  $('loadFile').addEventListener('change',e=>{const f=e.target.files[0];if(!f)return;const rd=new FileReader();
    rd.onload=async()=>{try{const sc=JSON.parse(rd.result);const r=await api('/api/load_scenario',sc);WORLD=r.layers;
      CATCOL={};CATICON={};WORLD.legend.forEach(c=>{CATCOL[c.id]=c.color;CATICON[c.id]=c.icon;});buildBackground();render(r.view);}catch(err){alert('Bad scenario file');}};rd.readAsText(f);});
  bindCanvas($('cvBase')); bindCanvas($('cvDss'));
  document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
    if(e.code==='Space'){e.preventDefault();playToggle();}
    else if(e.code==='ArrowRight'){stop();viewIndex<count-1?goto(viewIndex+1):goStep(1);}
    else if(e.code==='ArrowLeft'){stop();if(viewIndex>0)goto(viewIndex-1);}});
  $('btnHelp').addEventListener('click',()=>{$('helpBody').innerHTML=HELP;$('help').classList.remove('hidden');});
  $('helpClose').addEventListener('click',()=>$('help').classList.add('hidden'));
  $('help').addEventListener('click',e=>{if(e.target.id==='help')$('help').classList.add('hidden');});
  doReset();
});
