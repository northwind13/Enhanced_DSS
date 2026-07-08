# DisasterAware

Enhanced Decision Support System for Wildfire Disaster Response and
Management. Grid tabanli, ayrik zamanli hibrit yangin yayilim motoru
(tez Chapter 4 + Appendix A-C) ve uzerinde calisan tek ana uygulama.

## Calistirma

    run_dashboard.bat        DisasterAware ana uygulamasi (tek giris noktasi)

Ilk kurulum: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`

## Klasor yapisi (self-descriptive)

```
03_Codes/
  run_dashboard.bat        ana uygulamayi baslatir
  requirements.txt         bagimliliklar
  app/                     DisasterAware arayuzu (Streamlit)
    streamlit_app.py         tum sayfalar (Simulation, Map editor, Data
                             layers, Parameters, GIS import, Validation,
                             System Description)
    system_description.py    System Description sayfasi (tam matematiksel
                             dokumantasyon: mimari, simulator, DSS, maliyet)
  disaster_phyengine/      fizik motoru: gecis operatoru Phi, Rothermel ROS,
                           bastirma, siddet, maliyet, rewind, GIS import,
                           senaryolar, dogrulama metrikleri
  validation/              gercek veriyle dogrulama (hindcast)
    auto_validate.py         cekirdek: veri indirme + kor kosu + skorlar
                             (Validation sayfasi bunu kullanir; CLI da olur)
    cache/                   indirilen gercek veri, vaka bazli (git disi)
    runs/                    kosu arsivleri: report.json, log, haritalar,
                             kareler (git disi)
  tests/                   pytest suiti (27 test): motor + dogrulama
```

## Ana uygulamanin sayfalari

- **Simulation** - yangini kosturma; kosullar (ruzgar pusulasi, EMC nem,
  adim suresi), atesleme, rewind, J-terimli maliyet paneli.
- **Map editor** - harita uret/boyutlandir; yakit/yol/asset/yukselti boyama
  (2D canvas; 3D salt onizleme).
- **Data layers** - System Description ile ayni alanlar + operasyonel
  teshisler (Byram).
- **Parameters** - tum parametreler; literatur varsayilanlari (Anderson
  1982, Scott & Burgan 2005, Rothermel 1972).
- **GIS import** - gercek DEM/yakit rasteri ile harita.
- **Validation** - gercek yanginlara karsi otomatik hindcast (Copernicus
  DEM + ESA WorldCover + ERA5 + NASA FIRMS), canli izleme, ruzgar
  belirsizligi ansambli, kosu arsivi. Yontem/metrik/protokol sayfanin
  icindeki basliklarindadir.
- **System Description** - modelin tam matematiksel dokumantasyonu
  (arama + icindekiler agaci).

## Test

    .venv\Scripts\python -m pytest tests -q
