# Tez güncelleme promptu — Layer 3/4 sayısal tanımlar (koddan teze)

Aşağıdaki eklemeleri DISASTERAWARE_PhDThesis_v2 dokümanına işle. Kod tarafı
(03_Codes/dss/fuzzy.py, concepts.py, rules.py) bu değerlerle implement
edildi; tez ile kod birebir aynı sayıları kullanmalı. Metinler İngilizce,
tezin mevcut üslubuyla (we/I yok, emdash yok) yazılmalı. [REF] gerekmiyor.

## 1. Figure 2.3'ün sayısal karşılığı (Chapter 2, Figure 2.3 paragrafına ek)

Figure 2.3 beş terimli trapez bölüntüyü gösteriyor ama (a,b,c,d) değerleri
metinde yok. Figürün hemen ardına şu içerikte bir cümle + tablo ekle
(0.62 → M 0.53 / H 0.47 çalışan örneğini birebir üreten geometri):

"Each term is a trapezoid (a, b, c, d) on the normalized universe with
uniform centers, cores of half-width 0.05 and supports of half-width 0.20;
the end terms saturate outward. The parameters are listed in Table 2.X."

Table 2.X Five-term partition parameters (normalized universe)

| Term | a | b | c | d |
|---|---|---|---|---|
| VL | (saturated) | 0.00 | 0.05 | 0.20 |
| L | 0.05 | 0.20 | 0.30 | 0.45 |
| M | 0.30 | 0.45 | 0.55 | 0.70 |
| H | 0.55 | 0.70 | 0.80 | 0.95 |
| VH | 0.80 | 0.95 | 1.00 | (saturated) |

Not: Fig 2.3 drawio çizimi bu değerlerle uyumlu değilse çizim de
güncellenmeli.

## 2. Eq (40) agregasyon ağırlıkları ω (Chapter 4, Layer 3 bölümüne tablo)

Eq (40) ağırlıkların konsept başına 1'e normalandığını söylüyor ama sayısal
değerler verilmemiş. Eq (40) açıklamasının ardına şu tabloyu ekle
("The design weights are listed in Table 4.X; they are the initial values
that the adaptation loop of Layer 4 may later perturb." gibi bir cümleyle):

Table 4.X Concept hierarchy: inputs and aggregation weights

| Concept (level) | Inputs (weight) |
|---|---|
| Fire severity (1) | fire intensity (0.60), spread potential (0.40) |
| Spread hazard (1) | spread potential (0.45), weather severity (0.35), ignition proximity (0.20) |
| Fuel hazard (1) | weather severity (0.40), fuel load (0.60) |
| Asset value (1) | asset exposure (1.00) |
| Crew reachability (1) | resource accessibility (0.55), access and road status (0.45) |
| Logistics support (1) | suppression availability (1.00) |
| Fire threat level (2) | fire severity (0.40), spread hazard (0.40), fuel hazard (0.20) |
| Asset exposure risk (2) | spread hazard (0.45), asset value (0.55) |
| Suppression feasibility (2) | crew reachability (0.60), logistics support (0.40) |
| Intervention urgency (3) | fire threat level (0.45), asset exposure risk (0.35), temporal urgency (0.20) |
| Evacuation pressure (3) | asset exposure risk (0.40), inverted access and road status (0.30), temporal urgency (0.30) |
| Operational priority (4) | fire threat level (0.50), asset exposure risk (0.50) |

Not: evacuation pressure satırındaki "inverted access and road status"
(1 − z8) metinde bir cümleyle tanımlanmalı: kapalı/yetersiz yol ağı tahliye
baskısını YÜKSELTİR.

## 3. Eq (41) sönüm katsayısı ρ (Chapter 4, Eq 41 açıklamasına)

ρ ∈ [0,1) deniyor ama değer verilmemiş. Şu cümleyi ekle:

"The decay factor is set to ρ = 0.9 per decision cycle, so a concept that
ceases to be observed loses about one third of its effective activation
within four cycles and fades below the lowest linguistic core within
fifteen."

## 4. Kural dilbilgisi tutarsızlığı (Chapter 4, Rule Base bölümü + Appendix D)

Metin "Intervention rules are written exclusively over the gated
activations of the five decision concepts" diyor; ancak Appendix D'deki R4
("access status is low") ve R7 ("temporal urgency is high") birer FEATURE
okuyor. İki seçenekten biri uygulanmalı; kod (a) seçeneğiyle yazıldı:

(a) Önerilen: "exclusively" cümlesini şöyle yumuşat: "Intervention rules
are written over the gated activations of the five decision concepts;
where an operational regime requires it, a rule may additionally read a
gated feature as an auxiliary antecedent, as R4 and R7 of Appendix D do."

(b) Alternatif: R4/R7'yi yalnız konsept okuyacak şekilde yeniden yaz
(bu durumda kodda SEED_RULES da değişmeli; bana haber ver).

## 5. Çıktı bölüntüsü (Chapter 4, Rule Base bölümüne bir cümle)

Müdahale şiddetlerinin evreni tanımlanmalı:

"Each intervention intensity is defuzzified on the same five-term
partition over the normalized [0, 1] intensity universe, with the centroid
of Eq. (4); an intensity of zero means the intervention is not ordered."

## 6. Kontrol listesi (diğer chat için)

- [ ] Table 2.X partition parametreleri eklendi, Fig 2.3 ile tutarlı
- [ ] Table 4.X ω ağırlıkları eklendi, Eq (40) referansı verildi
- [ ] ρ = 0.9 cümlesi Eq (41) açıklamasına girdi
- [ ] "exclusively" cümlesi (a) seçeneğiyle güncellendi (veya (b) seçildi
      ve kod ekibine bildirildi)
- [ ] Çıktı bölüntüsü cümlesi eklendi
- [ ] Kod ile tezin aynı sayıları kullandığı doğrulandı
      (03_Codes/dss/fuzzy.py: default_partition; concepts.py: HIERARCHY,
      RHO_PERSIST; rules.py: SEED_RULES)
