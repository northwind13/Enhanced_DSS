# GODA makalesi devir notu (kısaltma ve son düzenleme)

Bu not, IEEE Systems Journal makalesinin kısaltma ve son düzenleme aşamasını başka bir sohbette sürdürmek için hazırlandı. Bilimsel içerik tamamlandı; yeni deney yapılmayacak. Kalan iş metni 12 sayfaya indirmek ve dört belgeyi tutarlı tutmak.

Son durum: repo `github.com/northwind13/Enhanced_DSS`, dal `main`, commit `f3328bd` (31 Ekim 2026, üçüncü bağımsız hakem düzeltmeleri dahil).

---

## 1. Dosyalar

| Belge | Yol |
|---|---|
| Makale | `02_Article/ieee_systems/DisasterAware_IEEESystems.docx` |
| Ek dosya | `02_Article/ieee_systems/DisasterAware_Supplementary_Material.docx` |
| Kapak mektubu | `02_Article/ieee_systems/DisasterAware_CoverLetter.docx` |
| Başlık sayfası | `02_Article/ieee_systems/DisasterAware_TitlePage.docx` |
| İstatistik tablosu (Holm, 65 satır) | `03_Codes/experiments/benchmark/revision/holm_tables_vii_viii.csv` |

Başlık (dört belgede aynı): **Forecast-Gated Search and Persistent Admission of Composed Interventions for Wildfire Decision Support**

Yazarlar: Çağlar Akman, Kemal Leblebicioğlu (METU, Elektrik ve Elektronik Müh.). Kurum olarak yalnızca METU geçer, başka kurum adı yazılmaz.

---

## 2. Yazım kuralları (proje kuralları, değişmez)

1. Makale, ek dosya ve kapak mektubu İngilizce. Sohbet Türkçe.
2. Uzun tire (em dash) kullanılmaz. Yerine virgül, noktalı virgül ya da iki cümle.
3. "we", "I", "our" kullanılmaz. Edilgen ya da özne olarak "this study", "GODA", "the closed layer" vb.
4. Sade, doğrudan, insan elinden çıkmış gibi yazım. Yol haritası cümleleri ("In this section we...") ve abartılı sıfatlar yok.
5. Atıf biçimi IEEE: `[3], [4]` (köşeli parantezler ayrı); `[3, 4]` biçimi kullanılmaz. Sayfa aralıklarında kısa tire (en dash).
6. Özet en fazla 250 kelime (şu an 250). Kapak mektubu tek sayfa (şu an 524 kelime).
7. IEEE Systems Journal sınırı: düzenli makale 12 sayfa. LibreOffice çıktısında şu an 17 sayfa (Word'de biraz az çıkabilir).

---

## 3. Word dosyasıyla çalışırken teknik uyarılar

Bu kısım önemli; aksi halde dosya bozulur ya da atıflar kaybolur.

1. **Atıflar Word alanıdır** (Citation content control, `CITATION Etiket` alanları) ve kaynakça bir `BIBLIOGRAPHY` alanıdır. Metindeki `[12]` gibi numaralar düz metin değildir. Bir cümle silinirse içindeki atıf da silinir; o kaynak başka yerde geçmiyorsa kaynakçada atıfsız kalır. Atıfları düz metin olarak yeniden yazma.
2. **Tablo ve şekil numaraları SEQ alanıdır**, metindeki "Table V" gibi atıflar REF alanıdır. Tablo silinir ya da taşınırsa Word'de Ctrl+A ve F9 ile güncelle.
3. **Denklemler Word denklemidir (OMML)**. Çok satırlı denklemler harf alt numaralıdır: (13a–d), (15a–c), (16a–f), (18a–d), (19a–d), (21a–c). Metindeki "(13)" gibi atıflar grubu gösterir.
4. Önerilen yöntem: ChatGPT metni ve önerilen silmeleri versin, değişiklikleri Word'de izli değişiklik açıkken sen uygula. Python ile docx yeniden yazılırsa alanlar düz metne dönüşebilir.
5. Kısaltma bitince Word'de: Ctrl+A, F9 (alanlar), References sekmesinden "Update Citations and Bibliography", sonra PDF al ve sayfa sayısına bak.

---

## 4. Kilitli iddialar ve sayılar (hiçbiri değişmeyecek)

Kısaltırken bu sayılar özet, V-C, VI ve kapak mektubunda aynı kalmalı. Bir yerde değişirse hepsinde değişmeli.

### 4.1 Ana karşılaştırmalar (150 eşli sentetik olay, S1–S5, tohum 201–230)

| İddia | Sayı | İstatistik |
|---|---|---|
| GODA, en güçlü kapalı sözlük aramasına (evrimsel arama, yalnız altı aile) karşı, eşit tahmin bütçesinde | yanan alan −32 ± 16%, toplam maliyet −25 ± 10% | Holm sonrası p < 0.001 |
| GODA, rastgele kapalı sözlük aramasına karşı | −35 ± 17% | Holm sonrası p < 0.001 |
| GODA, aynı gramerde evrimsel aramaya (12 × 3) karşı | −24 ± 18% | Holm sonrası p = 0.09, **anlamlı değil** |
| GODA, aynı gramerde rastgele tam aramaya karşı | −28 ± 20% | Holm sonrası p = 0.047, anlamlı |
| GODA, aramasız kapalı katmana (Closed) karşı | −61 ± 14% (75.4 → 29.1 ha), 131 daha iyi / 10 eşit / 9 daha kötü | p < 0.0001 |
| Independent gate futures (dağıtım için önerilen ayar) | −54% (34.0 ha) | |
| Coating fiyatlı ve erişimi sınırlı (K27) | −54 ± 13% (kendi Closed referansı 77.8 → 36.0 ha), %81 sönmüş | |
| Episod sonunda yanan hücre kalmayan olay oranı | %52 → %91 | Wilson aralığı |
| Üç model hatası birlikte | −30 ± 9% (82.3 → 57.4 ha), %55 → %65 | |
| Büyük alan, 3.6 × 2.7 km, 30 dünya | 309 → 134 ha (−57 ± 26%), %33 → %70 | |
| Sabit optimizörde gramerin tek başına katkısı | %9 ± 14 (rastgele), %11 ± 14 (evrimsel), yani Closed alanının 5 ve 6 puanı. "9 to 11 points" yazılmaz, birim yüzde | **anlamlı değil** |
| Kapalı sözlük aramaları kazancın ne kadarını alıyor | rastgele %41 (üçte iki), evrimsel %43 (%70) | |
| Rastgele önerici, bileşik gramer, tam arama | −46% (dörtte üç) | |
| LLM yalnız aday vs küçük rastgele kontrol (6 tahmin) | 14 ± 11 puan (45.5% vs 31.6%) | Holm sonrası p = 0.083, anlamlı değil |
| El yazımı taktikler arama tohumu olarak | −31 ± 12%; GODA ona karşı −44 ± 16% | GODA marjı anlamlı |
| Kalıcılık ablasyonu (nesne bir döngü sonra geri çekilir) | +2 ± 5%, kazanç tamamen kaybolur | p = 0.82 |
| Kapılar tek gelecek, yalnız G5 | −55 ± 13%; zararlı kabul %14 vs %10 | fark **anlamlı değil** |

### 4.2 Doktrin ve aktarım

| İddia | Sayı |
|---|---|
| 40 kurallı kapalı kontrol | −11 ± 8% (Holm p = 0.06, anlamlı değil) |
| Altı el yazımı taktik, yazıldığı gibi | −9 ± 6% |
| Aynı taktikler tahminle ayarlanmış (K30b) | −6 ± 7%, yazılandan farkı yok; seçimlerin %74'ünde varyantlar eşit |
| Kayıtlı dört taktik, 75 ayrılmış dünyada | −50 ± 15% (aynı 75 dünyada GODA −54 ± 18%) |
| S6–S8 (90 yeni dünya), kayıtlı + kapalı | −43 ± 11% |
| S6–S8, açık katman sürdürülünce ek | −28 ± 10% |

### 4.3 İstatistik çerçevesi

- Tek aile, 65 karşılaştırma: ana karşılaştırma, Tablo VI, VII ve S7'nin her satırı ve metinde p ile verilen sekiz ikili marj. İki taraflı eşli işaret çevirme testi (10^6 permütasyon), Holm düzeltmesi. 55 karşılaştırma 0.05 altında.
- Yarı genişlikle verilen farklar betimseldir, test edilmez.
- Ana karşılaştırma Tablo VI'daki GODA satırıdır (aile 66 değil 65). 65 karşılaştırmanın ham p, Holm sırası ve düzeltilmiş p değeri ek dosyada Tablo S10'da. p değerleri kesin işaret çevirme testinden; çarpık farklarda normal yaklaşımla aralıklardan hesaplanan değerden küçük olabilir (bir hakem bunu hata sandı, kontrol edildi, doğru).
- Karma etkiler modeli (aile rastgele kesişim): yanan alan −46 ha (SE 18, p = 0.010); beş aileyle küçük örneklem testi p ≈ 0.06, yani aileler arası genelleme kurulmuş değil. Dünya düzeyindeki eşli test ana analizdir.
- Dünya başına medyan değişim: −21 ha (−49%).
- p değerleri yanan alan farkını test eder.

### 4.4 Diğer sabit sayılar

- 878 model çağrısı, 5268 model adayı, 105 857 aday incelendi, 359 nesne kabul edildi. Model Claude Opus 5, 20–30 Ekim 2026.
- Tahmin bütçesi: arama 36 + en çok 3 finalist. Loglarda GODA çağrı başına 34.5, rastgele ve evrimsel arama 39.
- Hindcast (4 yangın): POD 0.81–0.91, Sørensen 0.80–0.90, cephe uyumu 0.84–0.90, FAR 0.11–0.22, varış hatası 0.6–3.8 saat. Eşit alanlı disk 0.27–0.50.
- Zamanlama: karar hesabı medyan 173 ms, 90. yüzdelik 526 ms, en çok 942 ms; LLM'li döngüler 90. yüzdelik 16 s, en çok 55 s.
- Kaplama etkinliği taraması: 0.75 / 0.5 / 0.25 / 0 → %55 / 46 / 31 / 31.
- Etki çıkarma: coating içeren nesneler çıkarılınca −21% (tam tekrar −55%, 150 dünya); kazancın yaklaşık üçte ikisi coating'li nesnelerde.
- S2, Closed yanan alanının %49'u; S2 hariç kazanç değişmiyor (%61 ± 14, 120 dünya).
- Kaynakça 50 kaynak.

---

## 5. Kaybolmaması gereken çekinceler

Kısaltırken tekrarlar silinebilir, ama her biri **en az bir yerde** (tercihen VI'da) kalmalı. Silinirse aynı hakem itirazları geri gelir.

1. Gramerin tek başına payı ve önericinin (LLM) payı ayrışmıyor; desteklenen iddia ikisinin birlikte en güçlü kapalı sözlük optimizörüne karşı kazancı.
2. LLM'in aynı gramerdeki evrimsel aramaya üstünlüğü kanıtlanmadı. Model yoksa rastgele önerici önerilen varsayılan ayar; LLM isteğe bağlı aday kaynağı.
3. Dayanıklılık kapalı katmana karşı ölçüldü; güçlü aramalara karşı marjlar yalnızca nominal modelde ölçüldü.
4. Etki operatörleri (coat, clear, ignite vb.) modelleme varsayımı; operasyon verisiyle kalibre edilmedi. Duyarlılık testleriyle sınırlandı.
5. Hindcast yalnızca iz şeklini ve konumunu sınar; yayılma hızını, müdahale etkisini ve kararları sınamaz.
6. Ayrı bir yaşam güvenliği garantisi yok. Nüfus maruziyetinin fiziksel maliyetteki ağırlığı küçük ve alan kazancı onu telafi edebilir. Tahliye ve uyarı emirleri kısıtlanmaz, ama bu zamanında tahliyeyi garanti etmez.
7. "Extinguished" operasyonel söndürme değildir; episod sonunda yanan hücre kalmayan olaydır. Alan kenarında sönenler de dahildir ve ayrı loglanmadı.
8. Kalıcılık ablasyonu emrin sürmesini ve kuralın yeniden kullanımını birlikte kaldırır; ikisi ayrılmadı.
9. Dağıtık işletim, iletişim gecikmesi, operatör değerlendirmesi ve veri asimilasyonu test edilmedi; karar çekirdeği tek süreçte değerlendirildi.
10. Kalıcı kör bölge, gecikmeli durum ve yanlış tespit test edilmedi; test edilen hata sınıfları parametre ve yapı hataları, yeniden çekilen kayıplar ve bozuk brief.
11. İki gölge gelecek yalnızca ember çekimlerinde ayrışıyor (medyan fark %0.2, işaret uyumu %99); G6 yalnızca tek ember yoluna uydurmaya karşı korur. Dağıtım için independent futures önerilir.
12. Tanınmış optimizerlar (MCTS, CMA-ES) denenmedi; aynı bütçede rastgele ve evrimsel aramalar onların yerini tutuyor.
13. Güven kapılaması (ikinci sınır) tarif edildi ama ablasyonu yapılmadı.
14. Kalite testinde puanlanan dört kavram: fire threat, asset exposure, intervention urgency, evacuation pressure. Suppression feasibility puanlanmaz (önceki bir sürümde yanlış yazılmıştı).

---

## 6. Geri getirilmemesi gereken eski ifadeler

Bunlar önceki turlarda hakemlerin itirazıyla kaldırıldı; tekrar yazılmamalı.

- "The gain rests on ... not on the proposer" (kazanç önericiye değil). Yerine: kazanç tahmin kapılı aramada ve yürürlükte kalan nesnelerde; gramer ve öneri kaynağının ayrı payları kurulmadı.
- "the composed grammar, not the lookahead alone, carries the difference"
- "life safety is not traded against area", "structure already guarantees"
- "the doctrine record carries the gain" (kalıcılık ablasyonundan yalnız doktrine bağlanan sonuç)
- "on reseeded futures" (özette), "two independently seeded futures" (G5/G6 için)
- "only the margin over the closed-vocabulary search survives"
- "For the random search a larger budget would not close the gap"
- "the gates bound/reduce the harmful admissions" (anlamlılık olmadan)
- "evaluated in simulation as one system" (kapak mektubunda)
- "affected population fell by 80%" özette ya da kapak mektubunda öne çıkarılmamalı
- "no reported comparison is left outside" (Holm ailesi için)
- "9 to 11 points" (doğrusu %9 ve %11, ya da 5 ve 6 puan)
- "does not rest on treating the worlds as independent"
- "strongest search" (doğrusu "strongest tested search")
- "open decision space" başlıkta ve özette ana çerçeve olarak (başlık değişti; metinde "bounded, compositional" niteliğiyle geçebilir)

---

## 7. Yapı (şu anki hali)

- **I. Introduction**
- **II. Related Work and Background** (A–D)
- **III. System Architecture** (A. Two Cores and the Interface, B. State, Transition, and Spread)
- **IV. The Decision Core** (A. Sources, Features, Confidence; B. Concept Space; C. Closed Layer; D. Open Layer; E. Local Agents and Global Coordination)
- **V. Results** (A. Experimental Design, B. Hazard Model Validation, C. Decision Quality and Robustness, D. Doctrine Transfer and Adaptation, E. Timeliness and Scalability)
- **VI. Discussion and Conclusion**
- Acknowledgment (yapay zekâ kullanım beyanı dahil), References (50), biyografiler

Tablolar: I tohum kuralları, II senaryolar, III el yazımı taktikler, IV hindcast, V ana sonuçlar (Closed / GODA / GODA coating priced + alt blok + senaryolar), VI arama ve kapı karşılaştırmaları, VII doktrin karşılaştırmaları.
Şekiller: 1 mimari, 2 kavram uzayı, 3 kapalı katman, 4 açık katman, 5 hindcast haritaları, 6 kabul edilen nesneler.

Ek dosya: Tablo S1–S10, Fig. S1 (zamanlama), Algorithm S1 (karar döngüsü).
- S7 model hatası ve işletim kısıtı satırları (Holm p ile)
- S8 tüm kontroller ve testler
- S9 aşama başına kullanılan maliyet ve tahmin sayısı
- S10 65 karşılaştırmanın tamamı (ham p, sıra, Holm p)

---

## 8. Kısaltma önerisi (yaklaşık 4–5 sayfa)

Sıra önerisi, en çok kazandırandan başlayarak:

1. **V-C** (yaklaşık 1600 kelime): Sayı yığınlarını tablolara bırak. Beş kısa paragraf kalsın: (a) ana sonuç ve eşit bütçeli referans, (b) arama ve öneri kaynağı karşılaştırmaları, (c) kapılar ve kalıcılık, (d) coating ve etki operatörleri, (e) model hataları ve büyük alan. Tablo VI'da olan sayıyı metinde tekrar etme.
2. **VI**: Çekinceleri tek paragrafta topla (bölüm 5'teki liste). Özetle ve V-C ile tekrar eden cümleleri sil.
3. **II-C ve II-D**: Her biri bir paragrafa insin. Atıflar korunmalı ya da kaynak kaynakçadan da çıkmalı.
4. **V-B (hindcast)**: Yarıya inebilir. Tablo IV ve Fig. 5 kalsın, replikasyon ve disk/elips karşılaştırması bir cümle olsun.
5. **IV-D, IV-E**: Uzun tanım paragrafları kısalabilir; denklemler kalmalı.
6. **V-A**: Model hatası testlerinin tarifleri Tablo S8'e bırakılabilir (zaten orada).
7. Fig. 2 ve Fig. 4 küçültülebilir; yazı boyutları okunur kalmalı.
8. Tablo III (el yazımı taktikler) gerekirse eke taşınabilir; taşınırsa tablo numaraları yeniden sıralanmalı.

Kesilmemesi gerekenler: Tablo V ve VI, özet, bölüm 5'teki çekinceler (en az bir yerde), Holm ailesi açıklaması, yapay zekâ beyanı, dipnottaki repo bağlantısı.

---

## 9. Hakem süreci özeti (cevap mektubu için)

Birbirini görmeyen birkaç hakem raporu alındı. Hepsi Majör dedi; ortak gerekçeler uzunluk, eşit bütçesiz %61 sunumu ve coating'in gücüydü. Yapılanlar:

| Konu | Cevap |
|---|---|
| %61 eşit bütçeli değil | Özet eşit bütçeli karşılaştırmayla açılıyor (%32, %24); %61 "aramasız kapalı katmana karşı" diye veriliyor |
| Güçlü baseline yok | Altı aile üzerinde evrimsel arama (K33) ve rastgele aile araması eklendi; MPC ayrımı II-C'de |
| Gramer ile efekt erişimi karışıyor | Sabit optimizörde gramer 9–11 puan, anlamsız; açıklama V-C'de |
| Coating aşırı güçlü | Fiyatlı ve erişimi sınırlı coating Tablo V'te üçüncü sütun (%54); etkinlik taraması (%55→31) |
| Oracle planlayıcı | Independent futures %54; dağıtım için önerilen ayar |
| Holm tutarlılığı | Aile 65, ikili marjlar dahil |
| Kalıcılık | Ablasyon (K28) eklendi; yorum emir ve yeniden kullanımın birlikte kaldırıldığı şekilde daraltıldı |
| El yazımı taktikler zayıf mı | Tahminle ayarlandı (K30b), sonuç değişmedi |
| Yaşam güvenliği | Garanti iddiası kaldırıldı |
| Kabul maliyeti ve tahmin bütçesi | Koddan doğrulandı, Tablo S9 |
| Model hatası, durum hatası, büyük alan, maliyet fonksiyonu, bütçe taraması | Hepsi koşuldu, Tablo V ve S7 |

Bilinçli olarak yapılmayanlar (cevap mektubunda gerekçelenecek):
- Kalıcı kör bölge, gecikmeli durum, iletişim gecikmesi testi (K32): kapsam dışı, sınırlama olarak yazılı.
- Büyük alanda 150 dünya: 30 dünya ile sınır gösterildi.
- MCTS, CMA-ES, RL tabanları: evrimsel ve rastgele aramalar aynı bütçede yerini tutuyor.
- Açık ağırlıklı ikinci bir LLM: LLM katkısı zaten iddia edilmiyor.
- Nüfus maliyeti ağırlığını artırmak: sınırlama olarak yazıldı.
- Efekt-erişim tabanı (K34): gramer iddiası zaten daraltıldı.
- Kalıcılık ablasyonunun ayrıştırılması (K28b): yorum daraltıldı.

---

## 10. Kalan işler

1. Makaleyi 12 sayfaya indirmek (bölüm 8).
2. Word'de alanları güncellemek, kaynakçayı yenilemek, PDF ile sayfa sayısını doğrulamak.
3. Fig. 1, 3 ve 5 başlıklarının PDF'te göründüğünü kontrol etmek; Fig. 3'te kalmış olabilecek eski sembolleri (α_min → μ_min, c → ĉ, ξ_f → ψ_f) düzeltmek.
4. `github.com/northwind13/GODA` reposunu public yapmak (kod, dünya tohumları, `proposals.jsonl` logları, Holm csv). Makale dipnotu, kapak mektubu ve başlık sayfası bu bağlantıya dayanıyor.
5. Son kontrol: bölüm 4'teki sayılar özet, V-C, VI ve kapak mektubunda aynı mı; bölüm 5'teki çekinceler en az bir kez geçiyor mu; bölüm 6'daki ifadelerden biri geri gelmiş mi; özet 250 kelimeyi aşıyor mu; atıfsız kaynak var mı.
6. Gönderim.

### Notasyon (metinde kullanılan, şekillerde de aynı olmalı)

| Sembol | Anlamı |
|---|---|
| χ_L, χ_m | sönme yükü ve nemi |
| Λ_k | biriken komşu etkisi (tutuşma için eşik Θ_ign = 0.125) |
| Ψ_k | adım başına komşu etkisi |
| ψ_f | özellik çıkarım haritası (ξ yalnız tohumlar için) |
| μ_min | kapsama tabanı (eski α_min) |
| ĉ_i^(l) | ara kavram düğümleri |
| ζ | uyarlama adım boyu |
| ℓ_in, ℓ_out | bant iç ve dış yarıçapı |
| q_t, s_t | eklenen terimin çekirdek ve destek yarı genişliği |
| a_t, b_t, c_t, d_t | terim köşeleri |
| E | bileşik müdahalenin cümle kümesi |
| ϑ_c | ölçeklenmiş cümle miktarı |
| Γ_DSS | bölgesel alanları birleştiren operatör |
| 𝓕_in, 𝓕_DSS, 𝓒_n | girdi ve karar alanları, kavram girdi kümeleri (script) |
