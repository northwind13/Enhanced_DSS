from PIL import Image, ImageDraw, ImageFont
import numpy as np, sys
# per-case window (cells) and optional anchor shift; output tiles all 600 px
CASES=[("canakkale2023","Sc1 Çanakkale 2023",200,(0,0)),("manavgat2021","Sc2 Manavgat 2021",280,(50,30)),("marmaris2021","Sc3 Marmaris 2021",150,(0,0)),("milas2021","Sc4 Milas 2021",200,(0,0))]
OV=(46,160,67); SIM=(200,55,44); OBS=(58,110,220)
def mask(a,c,t=40): return (np.abs(a.astype(int)-np.array(c)).sum(2)<t)
T=600
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",26)
tiles=[]
for f,lab,W,(dx,dy) in CASES:
    im=Image.open(f+".png").convert("RGB"); a=np.array(im)
    m=mask(a,OV)|mask(a,SIM)|mask(a,OBS); ys,xs=np.where(m); cy,cx=int(ys.mean())+dy,int(xs.mean())+dx
    x0=min(max(cx-W//2,0),a.shape[1]-W); y0=min(max(cy-W//2,0),a.shape[0]-W)
    crop=im.crop((x0,y0,x0+W,y0+W)).resize((T,T),Image.NEAREST)
    d=ImageDraw.Draw(crop); tw=d.textlength(lab,font=font); d.rectangle((0,0,tw+16,38),fill="white"); d.text((8,4),lab,fill="black",font=font)
    # scale bar: 5 km = 5000/90 cells
    px=5000/90*T/W; d.rectangle((T-px-14,T-26,T-14,T-18),fill="white",outline="black"); d.text((T-px-14,T-56),"5 km",fill="white",font=font)
    tiles.append(crop); print(f,W,x0,y0)
g=6; LH=48
out=Image.new("RGB",(2*T+g,2*T+g+LH),"white")
for i,t in enumerate(tiles): out.paste(t,((i%2)*(T+g),(i//2)*(T+g)))
d=ImageDraw.Draw(out); f2=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",24)
x=10; y=2*T+g+10
for c,l in [(OV,"overlap"),(SIM,"simulated only"),(OBS,"observed only")]:
    d.rectangle((x,y,x+30,y+28),fill=c,outline="black"); d.text((x+40,y),l,fill="black",font=f2); x+=40+d.textlength(l,font=f2)+40
out.save(sys.argv[1] if len(sys.argv)>1 else "fig5_try.png"); print(out.size)
