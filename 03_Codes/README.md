# DisasterAware Simulation Study

Section IV (Simulation Test) icin yangin karar-destek simulasyonu.
Tezdeki yangin yayilma modelini (grid hucresel otomat, ruzgar anizotropili
Rothermel-tipi yayilma hizi) ve Section III'teki kavram tabanli bulanik DSS'i
uygular; baseline ile DisasterAware'i karsilastirir ve makaledeki tum
Section IV figurlerini uretir.

## Hizli baslangic (Windows)

En kolay yol: **`run_all.bat`** dosyasina cift tikla.

Bu betik sirasiyla:
1. Python'u bulur (`python` ya da `py`); yoksa uyarir.
2. `numpy` ve `matplotlib` kutuphanelerini **yalnizca eksikse** kurar (varsa atlar).
3. `exp_main.py`, `exp_sens.py`, `exp_maps.py` betiklerini sirayla calistirir.
4. Uretilen `fig_sim_*.png` figurlerini `figures\` klasorune tasir.

Python kurulu degilse python.org'dan kurun ve kurulumda
**"Add Python to PATH"** secenegini isaretleyin.

## Elle calistirma (istege bagli)

    cd D:\repos_github\Enhanced_DSS\03_Codes
    python -m pip install numpy matplotlib
    python exp_main.py     # nominal karsilastirma, olceklenme, kural-azaltma benchmark
    python exp_sens.py     # duyarlilik: gurultu, esik, kapasite, bolge sayisi
    python exp_maps.py     # ortam katmanlari + uzaysal anlik goruntuler

Not: `exp_sens.py`, `exp_main.py` tarafindan uretilen `results.json` dosyasini
okur; bu yuzden once `exp_main.py` calismalidir (run_all.bat bu sirayi korur).

## Dosyalar

- `firesim.py`   Yangin CA cekirdegi: durum (yanma, yakit, yogunluk, atesleme zamani),
                 hibrit gecis operatoru, Rothermel-tipi yayilma, bastirma baglantisi.
- `dss.py`       DisasterAware DSS: ozellik cikarimi, 5-terimli bulaniklastirma,
                 dort kavram, kavram kural tabanlari (4 mudahale turu),
                 koordinasyon (kaynak normalizasyonu + izdusum), guven kapisi,
                 satisficing degerlendirme + fail-safe.
- `scenarios.py` Rastgele Monte-Carlo senaryolari (atesleme, ruzgar, yakit).
- `exp_main.py` / `exp_sens.py` / `exp_maps.py`  Deney betikleri.
- `run_experiments.py`  Tum suite (tek dosya, referans).
- `results.json` Tum sayisal sonuclar.
- `figures/`     Makalede kullanilan fig_sim_*.png gorseller.
- `run_all.bat`  Windows calistirma betigi.

## Temel sonuclar (20 senaryo ortalamasi)

- Yanan alan      %43.7 (baseline) -> %32.4 (DisasterAware)
- Varlik kaybi    89.6 -> 34.2  (-%62)
- Kural cikarimi  625 kural 24.7 ms vs 15625 kural 820 ms  (33x hizlanma)
- Gurultude zarif bozulma; sonuc bolge sayisindan bagimsiz.

## Gereksinimler

- Python 3.9+
- numpy, matplotlib  (run_all.bat eksikse otomatik kurar)
