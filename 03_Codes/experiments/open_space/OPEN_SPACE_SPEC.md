# Açık Müdahale Uzayı Deney Spesifikasyonu (IEEE Systems Journal makalesi)

Durum: v1.1, 2026-10-03. Faz A kodda tamamlandı (156/156 test geçiyor); karar: geri alma deneyinde üç temel aktüatör birlikte çıkarılır. Kod tabanı: `03_Codes` (dss/, disaster_phyengine/, experiments/).
Amaç: makalenin ana iddiasını, "karar katmanı, simülatörün çalıştırabildiği fiziksel ilkellerden çalışma zamanında yeni müdahaleler üretir ve bunları yalnızca simülatör onayıyla doktrine katar" cümlesini, tekrarlanabilir ve hakem karşısında savunulabilir deneylerle kanıtlamak. Tezden farklı olarak makale tek bu soruya odaklanır.

---

## 0. İddia ve yenilik düzeyleri

Makalede "yeni müdahale" dört düzeyde tanımlanır; iddia Düzey 2 ve 3 üzerine kurulur.

| Düzey | Nesne | Literatür karşılığı | Makaledeki yeri |
|---|---|---|---|
| 0 | Mevcut altı ailenin şiddeti | kapalı katalog (WFDSS, Dubey, Chandra) | referans, T_F5 |
| 1 | Makro: ailelerin ağırlıklı demeti | Levin composite alternative | yan ürün, "yeni" sayılmaz |
| 2 | Konumlu müdahale: etki × sektör × menzil × miktar cümlecikleri | Struss süreç kompozisyonu, Yeh & Chang prosedür sentezi | ana iddia |
| 3 | Yeni ara kavram (yeni antecedent ekseni) | Colorni & Tsoukiàs "bilinmeyen alternatif" | ikinci iddia |

Savunulan çerçeve: "doğrulanmış kapalı bir taban (7 etki, 7 sektör) üzerinde açık kompozisyon". Sınırsız açıklık iddia edilmez. Hedef J sabit ve biçimsel kalır; yalnızca eylem uzayı açılır. Komutan kapıların üstünde kalır (geçersiz kılma yetkisi makalede bir cümleyle belirtilir).

---

## 1. Kampanyadan önce zorunlu düzeltmeler (Faz A) [TAMAMLANDI 2026-10-03]

Hepsi `dss/` içinde, birim testleri `tests/test_gates.py` (8) ve `tests/test_sectors.py` (7). Uygulanan haliyle:
- A1: `_cva` her iki tohumda tabanı koşuyor; `g3`/`g4` kayıtlarına `j_noaction` eklendi (adapt.py).
- A2: `trial_objective()` varsayılan `phys`; `DSS_TRIAL_OBJ=total` ablasyon.
- A3: `_band(core, rin, rout)` tek geometri; menzil her sektörde, rin=0 referans hücreleri içerir; `head/flank/rear` her boyuttaki yangın için tanımlı (`_fire_cosang`); doğrulama `0<=rin<=rout<=15`.
- A4: `_apply_ignite/_apply_coat/_apply_draft` ortak yardımcılar; `tactical_burn` = ignite@band[4,6]∧downwind∧diggable, `retardant_drop` = coat@band[3,5]∧downwind, `water_drafting` ve cümlecik `draft` aynı yardımcı, ikisi de havuz sınırından sonra (`phase="post"`). `clause_cells()` dışa açıldı (keşif eşdeğerlik ölçümü için).
- A5: brief'te ignite eşiği (0.50, fmoist<0.30) ve menzil tanımı; öğrenilmiş cümlecik makroları `describe_macro` ile listeleniyor.
- A6: revizyon döngüsü her revizyonu kendi kusuruyla yargılıyor.
- A7: kayıt defteri kaynağı `_genai_propose.proposer_name` ya da `prop["_source"]`'tan.
- A8: kampanya tutarlılığı Faz C'de uygulanacak (kod değişikliği yok).

Önceki metin (tasarım gerekçesi) aşağıda korunuyor.

### A1. G4 taban hatası (adapt.py ~1869-1873, 2025; `_cva`)
Şu an G4, ikinci tohumda adayı eylemsizlikle (`j0b` = no-action) karşılaştırıyor. Düzeltme: ikinci tohumda da mevcut kural tabanı koşulacak (`j0b` = base without candidate, seed 202). G5 için de aynı. `_cva` iki tohum için de (j_base, j_candidate, j_noaction) üçlüsü döndürecek. Etki: kabul sayıları düşebilir; tezdeki sayılar değil, yeni kampanya sayıları makaleye girer.

### A2. Deneme ölçütü (evaluate._trial_cost)
Varsayılan `DSS_TRIAL_OBJ=phys` olacak. Makale metni ile kod aynı şeyi söyleyecek. (Tez metni "physical" diyor; kodda `total` idi.)

### A3. Sektör çözümlemesi (actions._sector_mask, ~599-633)
- `assets`, `populated`, `at_fire` menzili yok sayıyor; farklı yazılmış iki aktüatör aynı emre derleniyor ve "yeni" sayımı şişiyor. Düzeltme: `assets` ve `populated` için menzil, hedef hücrelerin çevresindeki bant olarak uygulanacak (`_dilate(target, rout) & ~_dilate(target, rin)`, rin=0 ise hedefin kendisi dahil). `at_fire` için menzil yanan hücrelerden ölçülecek.
- `rin = 0` sessizce 1'e çekiliyor; `[0,1]` boş küme veriyor. Düzeltme: rin=0 geçerli, "hedefin kendisi dahil" anlamına gelir.
- Yangın 50 hücrenin altındayken head/flank/rear hepsi ring'e düşüyor. Düzeltme: rüzgâr vektörü her zaman mevcut (meteoroloji girdisi), yön testi yangın büyüklüğünden bağımsız yapılacak; centroid tek hücre olsa bile çalışır.
- Birim testler: her sektör × menzil kombinasyonu için hücre kümesi; iki farklı cümlecik asla aynı maskeye derlenmemeli (aynı etki ve sektörde farklı menzil → farklı küme).

### A4. Cümlecik `draft` ile temel `water_drafting` eşitliği (actions.apply_actuator_clauses, decision_to_resources ~951-994)
Cümlecik draft havuz sınırından önce uygulanıyor ve hava terimi yok, temel aktüatör sınırdan sonra uygulanıyor. Geri alma deneyi (E1) için ikisi aynı fiziğe bağlanmalı: tek bir `_apply_draft(cells, a, dist_water)` yardımcısı, ikisi de onu çağırır, ikisi de havuz sınırından sonra uygulanır. Aynı şekilde `ignite` cümleciği ile `tactical_burn`, `coat` ile `retardant_drop` aynı motor yoluna bağlanır (band seçimi farklı kalabilir, fizik aynı).

### A5. Brief doğruluğu (mission._effect_notes)
`ignite` için "no threshold" yazıyor; motorda 0.5 eşiği var. Düzeltilecek. Öğrenilmiş cümlecik makroları durum metninde tanımsız listeleniyor (adapt.py ~1617-1626); cümlecikler yazdırılacak.

### A6. Revizyon döngüsü (adapt.py ~1747-1794)
Onarım ve yeniden kapı kontrolü `while` dışında kalmış; `revision_budget > 1` ile revize öneri kontrol edilmeden model yeniden çağrılıyor. Döngü içine alınacak. Kampanyada `revision_budget = 2` kullanılacak (bir ücretsiz düzeltme hakkı).

### A7. Önerici etiketi
`src = "claude"` sabit (adapt.py ~1714). Önerici adı kayıt defterine önericinin kendisinden gelecek (bkz. Faz B).

### A8. Kampanya tutarlılığı
Tek havuz rejimi ve tek tohum profili: 5 kurallı profil, havuz `rcap × 0.6` (merdiven kampanyasıyla aynı), değerlendirme tohumları 201+. Geometri kampanyasının 40 kural / `ravail × 0.25` ayarı makalede kullanılmaz.

---

## 2. Önerici arayüzü ve üç önerici (Faz B)

### B1. Protokol
`dss/proposers.py` (yeni):
```python
class Proposer(Protocol):
    name: str
    def propose(self, ctx: ProposalContext, k: int = 1) -> list[Proposal]: ...
    def feedback(self, proposal: Proposal, verdict: GateVerdict) -> None: ...
```
`ProposalContext`: durum metni + yapısal alanlar (kavram değerleri, aktif kural tabanı, kelime dağarcığı, harita olanakları, havuz durumu, en yakın geçmiş vakalar, geometri tanısı). Metin yalnızca LLM için; arama önericileri yapısal alanları kullanır.
`Proposal`: mevcut `_GENAI_SCHEMA` sözlüğü + `source`.
`adapt._genai_propose` yerine `engine.proposer.propose(ctx, k)`; `k` aday döner, kapı zinciri sırayla dener, ilk kabul edileni alır (Fazlollahi: üretimi değerlendirmeden ayır, çoklu çeşitli aday).

### B2. Önericiler
1. `LLMProposer`: mevcut claude CLI yolu, aynı brief ve şema. `k=1` (maliyet), revizyon bütçesi 2.
2. `EvoProposer` (yeni): aynı gramer üzerinde evrimsel arama. Birey = (antecedent ≤3, consequent ≤3, isteğe bağlı 1 yeni nesne: makro | cümlecik aktüatörü ≤3 cümlecik | ara kavram ≤4 girdi). Uygunluk = gölge tahmin fiziksel maliyet düşüşü (tohum 101). Popülasyon 12, nesil 3, turnuva seçimi, mutasyon: etki/sektör/menzil/miktar/terim değişimi; çaprazlama: cümlecik takası. Ucuz kapılar (G1, G2, G2b, G2c, G2d, grounding) uygunluk hesabından ÖNCE eleme olarak uygulanır, böylece tahmin bütçesi geçersiz bireylere harcanmaz. Çıktı: en iyi `k=3` birey, kapı zincirine tam olarak (G3/G4/G5/G5b, tohum 202 dahil) tabi. Bütçe: çevrim başına en fazla 36 gölge tahmin.
3. `RandomProposer` (yeni): aynı gramerden tekdüze örnekleme, ucuz kapılardan geçen ilk `k=3` aday. Alt sınır.
4. `TemplateProposer`: mevcut `offline_proposer` (yalnızca referans; makalede "deterministic stand-in" olarak tek satır).

Bütçe eşitliği: her önerici için çevrim başına "gölge tahmin sayısı" sayılır ve raporlanır; LLM için ek olarak çağrı sayısı ve süre. Karşılaştırma hem eşit çevrim sayısında hem eşit tahmin bütçesinde yapılır.

### B3. Öğrenen kompozisyon tablosu
`dss/adapt.StageController`'a paralel ikinci tablo `CompositionPrior`: bağlam kovası (havuz dolu/boş, su var/yok, varlık ≤15 hücre var/yok, cephe büyüyor/duruyor) × (etki, sektör) çifti → kapıdan geçme oranı ve ortalama ΔJ. Güncelleme: her G3+ sonucu. Kullanım: EvoProposer başlangıç popülasyonunu bu tablodan örnekler; LLMProposer'a "bu bağlamda daha önce geçen/geçmeyen bileşimler" olarak brief'e eklenir (chronicle'ın yapısal karşılığı). Bu, "simülatörden öğrenme" iddiasının ölçülebilir karşılığıdır: tablo dolu/boş ablasyonu (E3).

---

## 3. Geri alma anahtarı ve senaryo aileleri (Faz C)

### C1. `DSS_WITHHOLD_ACTUATORS="tactical_burn,retardant_drop,water_drafting"`
Tek konfigürasyon anahtarı; etkilediği yedi nokta:
1. `rules.DISCOVERABLE_INTERVENTIONS` import-by-value → fonksiyon haline getir (`discoverable()`), adapt/resolve/evaluate çağrıyla okur.
2. `rules.ACTUATOR_LIBRARY` brief'e filtrelenmiş girer.
3. `adapt._GENAI_SCHEMA` çalışma zamanında üretilir.
4. `actions.decision_to_resources` geri alınan kanalı yok sayar.
5. `loop._decide_regions` koordinatör direktifi (`water_drafting = 0.6` zorlaması) geri alındığında kapanır. Not: bu direktif açık uzay deneylerinde her durumda kapatılmalı, çünkü kural dışı bir emir; makalede "doktrin dışı hiçbir emir yok" iddiasıyla çelişir.
6. `_situation_brief` öneri metni filtrelenir.
7. Şablon önerici filtrelenir.

### C2. Yeni senaryo aileleri (campaign5.SCENARIOS genişletmesi; registry'ye dönüştürülür)
Mevcut S1-S5 korunur (sürekliliği göstermek için). Açık uzayın yapısal olarak gerekli olduğu üç aile eklenir:

| ID | Harita | Ne test eder | Kapalı tabanın neden yetmediği |
|---|---|---|---|
| S6 Lakeside | River valley, cepheye ≤12 hücre göl, havuz `rcap × 0.4` | drafting'in kendiliğinden keşfi | doktrinde drafting yok, havuz tükeniyor; yalnızca draft kapasite ekleyebilir |
| S7 Fast WUI | Mediterranean coast, 12 yerleşim, rüzgâr 12 m/s, fmoist 0.06, 2 yakın tutuşma | ön koruma (coat@assets) keşfi | asset_protection kayıp azaltır ama tutuşmayı engellemez; yalnızca coat engeller |
| S8 Long incident | Rolling hills, 3 tutuşma, havuz `rcap × 0.4`, 18 saat tavan | tükenmiş havuzda ikame taktikler | wet/clear sıfır etki; draft ve ignite (karşı ateş) tek çıkış |

Her ailede su varlığı tohumdan bağımsız garanti edilir (S6, S7) ya da yokluğu garanti edilir (S8 için seçenek). `scenario.build_world`'e `water="lake_near_front"|"none"|"preset"` parametresi.

---

## 4. Deneyler (Faz D)

Ortak: 72×50, 30 m, 2 dk adım, 4 bölge, çevrim 12 dk, ufuk 24 dk, η 0.60, J_TH 0.35, 5 kurallı profil, soğuma 24 dk, 12 saat (S8: 18 saat), tohumlar 201-230 (30/koşul; LLM kolu için 201-215). Tüm koşular `DSS_TRIAL_OBJ=phys`, A1-A8 uygulanmış.

### E1. Geri alınmış doktrinin yeniden keşfi (ana kanıt)
- Kollar: `F5+Ev` (kapalı), `Open/Evo`, `Open/LLM`; her biri `withhold = {tactical_burn, retardant_drop, water_drafting}` ile (KARAR: üçü birlikte çıkarılır; senaryo başına tek hedef çıkarma yalnızca pilotta tanı amaçlı koşulur, makaleye girmez).
- Senaryolar: S6 (hedef: draft@at_fire/head), S7 (hedef: coat@assets/head), S8 (hedef: ignite@head/ring + clear).
- Ölçütler: (a) hedef taktiğin eşdeğerinin kabul edilip edilmediği (eşdeğerlik tanımı: etki eşleşir, sektör hedef kümesini ≥%60 örter, maskeler Jaccard ≥0.5 ile temel aktüatörün bandına denk), (b) keşfe kadar geçen süre (dk), (c) keşiften sonra fiziksel maliyet farkı, (d) yanan alan: kapalı vs açık vs "doktrinde var" (geri alınmamış) üst sınır.
- Beklenen tablo: satır senaryo, sütun kol; hücre "keşif oranı (30 tohumda), medyan keşif dakikası, ΔJ_phys". Şekil: üç senaryo için keşif zaman eğrisi (kümülatif).
- Bu deney "sistem gerçekten yeni müdahale üretiyor" iddiasının referanslı kanıtı: referans insan doktrini.

### E2. Kapalı ve açık uzayın yapısal farkı
- Kollar: `T0`, `F5`, `F5+Ev`, `Open/Evo`, `Open/LLM`, `Open/Evo+Lib` (temel aktüatörler açık, geri alma yok).
- Senaryolar: S1-S8.
- Ölçütler: yanan alan, söndürme payı, etkilenen nüfus, J_phys, J_total; 95% yarı genişlik; eşleştirilmiş farklar.
- Hipotez: S1-S5'te açık uzay küçük fark (tezdeki gibi), S6-S8'de büyük fark. Bu desen hakeme "açık uzay nerede gerekli" sorusunun cevabı.
- Greedy kolu (`greedy_baseline`, tohumlar 201+ ile yeniden) ve "tam saldırı" çalışma noktası (`satisficing off, share 1.0`) aynı tabloya girer; greedy'nin üretemedikleri ayrı satır: konumlu müdahale sayısı, tahliye emri sayısı, kısıt altında geri çekilme (fail-safe) sayısı.

### E3. Üç önerici karşılaştırması ve öğrenme ablasyonu
- Kollar: `Open/Random`, `Open/Evo`, `Open/LLM`, her biri `CompositionPrior` açık/kapalı (6 kol).
- Senaryolar: S6, S7, S8 (açık uzayın çalıştığı yerler).
- Ölçütler: öneri sayısı, kabul oranı (G3+ kabul / tahmin harcanan), kabul başına gölge tahmin, kabul edilen nesnenin ΔJ_phys'i (iki tohum), yeniden kullanım oranı (kabul edilen nesneyi sonra başka bir kuralın sipariş etmesi), çeşitlilik (farklı etki×sektör imzası sayısı), kabul edilen nesne türü dağılımı (kural/makro/cümlecik/kavram), duvar süresi.
- Eşit bütçe karşılaştırması: her kol için çevrim başına 36 gölge tahmin tavanı.
- Olası sonuçlar ve makaledeki okunuşu: LLM ≈ Evo → "önerici değiştirilebilir, garanti kapılardan gelir"; LLM > Evo → "dil modeli arama uzayını daraltıyor, kapılar yine karar veriyor"; Random ≈ diğerleri → gramer çok küçük, G5 marjı yükseltilmeli (bunu önceden bilmek gerekir, pilot 5 tohumla).

### E4. Kapı duyarlılığı ve dürüstlük testleri
- G5 marjı δv ∈ {1e-4, 1e-3, 1e-2}: kabul sayısı ve sonradan "zararlı" çıkan kabul oranı (kabul edilen nesneyi 5 yeni tohumda yeniden koş; ΔJ_phys > 0 ise yanlış kabul).
- G5b açık/kapalı: yalnızca eşlikli paketlerde fark; tezde hiç tetiklenmemişti, burada E3'ün LLM kolunda tetiklenmesi beklenir.
- A1 öncesi/sonrası (eski G4 vs düzeltilmiş G4) kabul sayısı: ekte bir cümle, "eski kapı x yanlış kabul veriyordu".

---

## 5. Çıktılar ve dosya düzeni

```
experiments/open_space/
  campaign.py        # tek giriş: --exp E1|E2|E3|E4 --arms ... --scenarios ... --seeds 201-230 --workers N
  proposers_bench.py # E3 bütçe sayacı
  report.py          # tabloları ve şekilleri üretir
  out/
    runs.csv         # ladder_runs şeması + proposer, withhold, prior, discovery_min, discovered(bool), n_sited, n_evac_orders
    proposals.jsonl  # her öneri: proposer, kind, gates, j_base/j_cand/j_noaction × 2 tohum, payload, reuse_count
    discovery.csv    # E1: scenario, arm, seed, target, discovered, t_discover_min, jaccard
    tables/ figs/
```
Makale tablo ve şekilleri: Tablo "scenario families", Tablo E1 keşif, Tablo E2 merdiven+greedy, Tablo E3 önerici karşılaştırması, Şekil keşif zaman eğrileri, Şekil kabul edilen cümlecik aktüatörleri (örnek 3-4 tanesi harita üstünde çizili), Şekil bütçe-kabul eğrisi (E3), Şekil G5 marjı duyarlılığı (E4).

---

## 6. Hesap bütçesi (tahmini, i9-10900K)

- Kapalı kollar: 45-100 s/koşu. E2 kapalı kısmı: 3 kol × 8 senaryo × 30 tohum ≈ 720 koşu ≈ 15 saat tek çekirdek; 8 işçi ile ~2 saat.
- Evo/Random kolları: çevrim başına ≤36 gölge tahmin; 12 saatlik olayda ~60 çevrim, adaptasyon tetiklenen ~%40'ı → ~860 tahmin/koşu, tahmin başına ~0.3 s → +4-5 dk/koşu. E1+E2+E3 Evo/Random toplamı ~ (3+3+4 kol) × 3-8 senaryo × 30 ≈ 1500 koşu ≈ 125 saat tek çekirdek, 8 işçi ile ~16 saat. İki gece.
- LLM kolu: 30-180 s/koşu ek, 15 tohum × (3+8+3) senaryo ≈ 210 koşu, seri (CLI oturumu), ~6-8 saat. Pilot önce 5 tohumla.
- Önce pilot: her deney için 5 tohum, sonuçlar yönü doğruluyorsa tam koşu.

---

## 7. Yapılacaklar sırası

1. Faz A (A1-A8) + birim testler (`tests/test_sectors.py`, `tests/test_gates.py`): 1-2 gün.
2. Faz B önerici protokolü + Evo + Random + CompositionPrior + testler: 2 gün.
3. Faz C geri alma anahtarı + S6-S8 + su kontrolü: 1 gün.
4. Pilot (5 tohum, tüm deneyler): 1 gece. Gözden geçirme, marj/bütçe ayarı.
5. Tam kampanya: 2-3 gece.
6. `report.py`, tablolar, şekiller; makale Bölüm V-VI yeniden yazımı (v3, izli).

---

## 8. Makaleye yansıyan iddialar (ve yansımayanlar)

Söylenebilecekler: (1) sistem, doktrinden çıkarılmış üç taktiğin eşdeğerini x/30 tohumda ortalama y dakikada yeniden keşfetti; (2) kapalı tabanın yapısal olarak yetersiz kaldığı üç ailede açık uzay yanan alanı %z azalttı, S1-S5'te fark küçük kaldı (dürüst rapor); (3) üç önerici aynı kapılarla karşılaştırıldı, garanti önericiden bağımsız; (4) öğrenen kompozisyon tablosu kabul başına tahmin maliyetini düşürdü (ya da düşürmedi, raporlanır).
Söylenemeyecekler: sınırsız açıklık; yeni fizik; LLM'in "anladığı"; sahada doğrulama.
