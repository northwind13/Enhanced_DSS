# Harici baseline koşusu: sonuçlar ve karar gerektiren iki bulgu

## Ne koşuldu
`03_Codes/experiments/greedy_baseline.py` (yeni, repoya yazıldı; `campaign5.py` de git geçmişinden geri getirildi). Politika: her 12 dakikalık çevrimde, yangının yakınındaki en yüksek değer taşıyan bölgelere havuzun tamamını değer oranında tahsis eden açgözlü dağıtım. Aynı simülatör, aynı dünyalar (S1-S5), aynı tohumlar (101-110), aynı kaynak havuzu, aynı aktüasyon (decision_to_resources). Kural yok, tahmin yok, kabul testi yok, tahliye/uyarı kanalı yok. 50 koşu tamamlandı; sonuçlar `experiments/out/greedy_runs.csv` + `greedy_curves.csv`.

## Sonuç (yanan alan, 10 dünya ortalaması, ha)
| | S1 | S2 | S3 | S4 | S5 | Ort |
|---|---|---|---|---|---|---|
| Eylemsizlik | 243.3 | 242.2 | 178.9 | 218.1 | 218.1 | 220.1 |
| **Greedy** | **13.0** | **120.7** | **11.3** | **42.6** | **42.6** | **46.0** |
| Statik 5 kural | 96.7 | 157.3 | 28.0 | 66.2 | 69.3 | 83.5 |
| Uyarlamalı | 84.7 | 154.6 | 30.5 | 54.6 | 66.3 | 78.1 |
| Full (best-of) | 78.3 | 145.1 | 24.4 | 53.4 | 53.5 | 70.9 |

## BULGU 1: Greedy, beş senaryonun beşinde de DSS'ten az yakıyor
Mekanik eşitlik doğrulandı: iki taraf da gerçek yangın maskesini görüyor (kampanyada sensör ağı yok), ikisi de 12 dk'da bir karar verip aradaki adımlarda emri sürdürüyor, ikisi de aynı havuzla sınırlı. Fark DSS'in kendi tasarımından geliyor: satisficing 0.35'te "yeterli" adayı kabul ediyor, dikkat payı ikincil bölgeleri 0.50'ye kırpıyor, TS harmanlaması 1.0 altı yoğunluk üretiyor. Bu maliyet modelinde erken tam yüklenme neredeyse optimal, çünkü tepki terimi yalnızca zaman integrali ve erken biten yangında integral küçük (greedy'nin j_resp'i bile çoğu senaryoda DSS'ten düşük).

Makaleye şu an eklenen çerçeve (V-D paragrafı + Fig. 14): dürüst rapor + mimarinin katkısını greedy'nin üretemediklerinde konumlandırma (denetlenebilir emir, kısıt altında fren, yaşam güvenliği kanalları; greedy 50 koşuda 0 tahliye emri verdi). Uyum tablosundaki "lowest in every scenario" ifadesi "lowest among the internal references" olarak yumuşatıldı. ANCAK abstract ve sonuçtaki "karar kalitesi" iddiası hâlâ ham alan üzerinden kurulu; hakem bu tabloyu görünce aynı soruyu soracak. Seçenekler:
- A) Bu çerçeveyle gönder (risk: karar-kalitesi iddiası zayıflar).
- B) DSS'in "tam saldırı" çalışma noktasını da koş (satisficing kapalı, dikkat payı 1.0): greedy ile eşitlenirse hikâye "greedy = mimarinin kendi agresif köşesi; kapılar x ha'ya mal oluyor ve karşılığında şunları veriyor" olur. En sağlam bilimsel çerçeve bence bu; koşusu kolay (kampanyaya bir arm).
- C) Tepki maliyetini literatüre göre yeniden kalibre et (ani tam seferberlik gerçek hayatta bedava değil); savunulabilir ama "kazanana kadar ayar" görüntüsü riskli.

## BULGU 2: Makaledeki Tablo V'in "Full" sütunu mevcut veriden türetilemiyor
Basılı sütun (22.8 / 84.8 / 11.4 / 35.7 / 39.3, ort. Jphys 0.090) tezden geliyor; ama repodaki ladder_runs.csv'nin HİÇBİR git sürümü (fin4 pilotu dahil) bu değerleri üretmiyor. T0 / TF5 / TF5+Ev sütunları mevcut CSV ile bire bir tutarken Full sütununu üreten F5EvAI koşuları sonradan üzerine yazılmış görünüyor. Mevcut veriden Full = 78.3/145.1/24.4/53.4/53.5 (ort. Jphys 0.139) çıkıyor; makale ise 39 ha ve 0.090 iddia ediyor. GÖNDERİMDEN ÖNCE ÇÖZÜLMELİ:
- ya basılı sütunu üreten kampanya yeniden koşulur (`python experiments/run_full_campaign.py`, ~saatler; greedy da aynı tohumlarla tekrarlanır, komut: `python experiments/greedy_baseline.py`),
- ya Tablo V, abstract ve sonuç mevcut kampanyaya göre yeniden yazılır (82% → ~%68, 0.090 → 0.139; Fig. 14 zaten bu veriyle tutarlı).

## Durum
- SJ_v2 hazır: yazar bloğu + kurum dipnotu (e-postalı) + Acknowledgment + fotoğrafsız biyografiler + Fig. 14 + V-D paragrafı; [TBD] kalktı. Tek kör hakemlik resmî sayfadan teyitli ("single-anonymous peer review").
- Sayfa: clean sürüm LO'da 12 (Word'de ~11 beklenir). Fig. 3 ve Fig. 7 düşürülürse ~10'a iner.
- Fotoğraflar bilinçli olarak eklenmedi (0.3 sf tasarruf; kabul sonrası eklenir).
