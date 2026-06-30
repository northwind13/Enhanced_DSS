# DisasterAware Wildfire Simulator

Tezin Chapter 4 (Simulation Framework) ve eklerinde tanimlanan hibrit yangin
yayilim modelini uygulayan, grid tabanli ve ayrik zamanli bir orman yangini
simulatorudur. Uzerine Decision Support System (DSS) kurulacak fiziksel
omurgayi olusturur.

## Ne yapar

- Grid uzerinde her hucre icin `s = (B, Fload, I, tau)` durumunu tutar:
  yanma durumu, kalan yakit, yangin siddeti, atesleme suresi.
- Rothermel tipi yayilim hizi (rate of spread), ruzgar yonlu anizotropik
  yayilim, egim ve aspect etkisi, yakit nemine bagli sonme.
- Bastirma (suppression) ile yakit azaltimi ve onleyici yakit azaltma
  (firebreak) etkisi.
- Cikti ve maliyet metrikleri: yanan alan, yanan orman, deger/asset kaybi,
  insan maruziyeti ve beklenen kayip, altyapi kaybi, bastirma maliyeti.
- Harita uzerine orman, asset, atesleme ve firebreak ekleme.
- Tum model parametrelerini ayarlama.
- Senaryo kaydetme/yukleme (JSON/YAML) ve GIS raster (DEM, yakit) import.

## Klasor yapisi

```
03_Codes/
  disasteraware/        simulasyon paketi (motor)
    config.py           parametreler (yakit modelleri, fizik, maliyet)
    layers.py           dis veri katmanlari (meteo, topo, fuel, value, resource)
    world.py            duzenlenebilir dunya/senaryo + editing API
    state.py            durum vektoru s=(B,Fload,I,tau)
    spread.py           Rothermel ROS + yonlu yayilim (Appendix A)
    suppression.py      bastirma -> yakit azaltma (Appendix B)
    intensity.py        yangin siddeti proxy (Appendix C)
    core.py             gecis operatoru Phi ve Simulator
    interaction.py      etkilesim operatoru Theta_UI (Bolum 4.2.4)
    costs.py            maliyet ve etki modeli
    viz.py              gorsellestirme yardimcilari
    gis.py              GIS raster import (rasterio opsiyonel)
    scenarios.py        hazir senaryolar
    io_utils.py         senaryo kaydet/yukle
  app/
    streamlit_app.py    interaktif dashboard
  tests/
    test_core.py        birim testleri
  examples/
    run_headless.py     terminal ornegi
```

## Kurulum

```bash
cd 03_Codes
pip install -r requirements.txt
```

## Kullanim

Dashboard:

```bash
streamlit run app/streamlit_app.py
```

Terminal ornegi:

```bash
python examples/run_headless.py
```

Python API:

```python
from disasteraware import Simulator, scenarios, compute_costs

world = scenarios.wui_interface()   # veya World.blank(...) ile sifirdan
sim = Simulator(world)
sim.run()                            # yangin sonene kadar
print(compute_costs(sim).to_dict())
```

Sifirdan dunya kurma ve duzenleme:

```python
from disasteraware import World, SimConfig, Asset, Simulator

w = World.blank(SimConfig(nx=120, ny=80, cell_size_m=30.0))
w.add_forest_patch(10, 10, 60, 60, fuel_type="pine_litter", load=1.0)
w.add_asset(Asset("Hastane", "critical", x=90, y=40, radius=2, value=1.0))
w.set_uniform_wind(speed=10.0, direction_rad=0.0)
w.add_ignition(x=20, y=40, step=0, radius=1)

sim = Simulator(w)
sim.run()
```

## Testler

```bash
cd 03_Codes
pytest -q
```

## Model haritasi (tez denklemleri)

| Bilesen | Denklem |
| --- | --- |
| Yanma durumu guncellemesi | 43 to 48 |
| Yakit kutlesi guncellemesi | 49, 68, 129 |
| Yangin siddeti | 51, 136, 137 |
| Atesleme suresi | 52 |
| Rothermel ROS | 123 to 128 (Tablo A.1) |
| Bastirma haritalamasi | 130 to 135 |
| Deger oncelik skoru V_prio | 55 |
| Etkilesim operatoru | 53, 54 (Tablo 4.1) |

## DSS entegrasyonu icin notlar

Simulator durumu disaridan dogrudan yazilmaz. Bir DSS, `sim.step()` cagrisina
`resource_override` (kaynak alani) ve/veya `extra_ignition` vererek yalnizca dis
girdi katmani uzerinden etki eder. Bu, Bolum 4.1 mimari ayrimini korur.
