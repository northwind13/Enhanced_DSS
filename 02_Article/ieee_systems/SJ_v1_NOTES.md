# SJ_v1 — ne yapıldı, ne kaldı

Taban: `ieee_systems/DisasterAware_Article.docx` (TSMC sürümü, 10 sayfa LO).
Çıktı: `SJ_v1_DisasterAware_tracked.docx` (değişiklikler işaretli) ve `SJ_v1_DisasterAware_clean.docx` (kabul edilmiş).
Ölçüm: clean sürüm LibreOffice'te **11 sayfa**, 8 076 kelime. Word'de tipik olarak bir sayfa daha az çıkıyor, yani beklenen 10 sayfa. Kesin sayıyı Word'de oku.

## Yapılanlar

**Başlık.** "DisasterAware: A Distributed Decision-Support Architecture with a Verified Open Intervention Space for Wildfire Response". "AI Enhanced" ve "Fuzzy Reasoning" başlıktan çıktı.

**Abstract (244 kelime, sınır 150-250).** Sistemlerin sistemi çerçevesiyle açılıyor. Dil modeli önerici abstract'tan çıkarıldı, yerine "interchangeable proposer" ifadesi kondu. "Life-safety orders are never attenuated" garantisi abstract'a girdi.

**Anahtar kelimeler (6 adet, sınır ≥5).** system-of-systems architecture; distributed decision support; open intervention space; verification and fail-safe design; human-supervised emergency response; wildfire management.

**Giriş.** İlk cümle yangın müdahalesini bir sistemler sistemi olarak tanımlıyor. Katkı beyanı paragrafı eklendi: hangi nesne tanımlanıyor, hangi invaryant korunuyor, ne ispatlanıyor, ne ölçülüyor; ve "bu bir çıkarım algoritması değil, sistem katkısıdır" ifadesi.

**Bölüm III.** Başlık "System Architecture and Interface Contracts", III.A "Entities, Responsibilities, and Interfaces" oldu. Mimari paragrafı beş katılımcılı sistemler sistemi olarak yeniden yazıldı (veri sağlayıcılar, simülasyon çekirdeği, N yerel ajan, çıkarımsız koordinatör, denetleyici komutan) ve arayüz sözleşmesi vurgulandı.

**Bölüm IV.** Başlık "Distributed Decision Layer and Its Guarantees". Sonuna **IV-H Guarantees** alt bölümü ve dört önerme eklendi: no-harm, life-safety invariance, coverage, bounded growth and termination. Her biri kısa ispatıyla.

**Bölüm VI (yeni).** "Discussion: Systems Lessons and Limitations": üç sistem dersi ve üç sınırlama. Sonuç bölümü VII oldu ve bozuk kalmış cümlesi onarıldı.

**Terminoloji.** "clause actuator" → "sited intervention" (kapsam dışı listesindeki kontrol sistemi "actuator" çağrışımı kaldırıldı). "closed-loop parameters" → "loop parameters". "open decision space" → "open intervention space". Metinde plant, controller, fuzzy control ifadeleri yok.

**Kısaltma.** Bölüm II, III ve IV'te açıklama ağırlıklı paragraflar sıkıştırıldı (yaklaşık 500 kelime), yeni eklenen içeriğe yer açmak için.

## Kalanlar (senin tarafında)

1. **Harici baseline.** Bölüm V-D'de köşeli parantezli bir `[TBD, to be completed after the baseline run]` paragrafı var. Aynı simülatörde, aynı tohumlarla, değer ağırlıklı açgözlü tahsis politikası koşulacak; sonuç Tablo V'e sütun olarak eklenecek ve o paragraf gerçek sonuçla değiştirilecek. **Bu marker kaldırılmadan gönderim yapılmamalı.**
2. **Önermelerin doğrulanması.** Dört önermenin öncülleri koddan teyit edilmeli, özellikle Önerme 4'teki "revision budget" ve "monotone test on a working copy" ifadeleri.
3. **Yazar bloğu.** Dosya hâlâ anonim. Systems Journal tek körse yazar adları, kurum dipnotu ve biyografiler eklenecek (yaklaşık +0,85 sayfa; fotoğrafsız biyografi 0,3 sayfa tasarruf ettirir).
4. **Sayfa hedefi.** 10 sayfaya inmek için figür düşürme gerekiyor: Fig. 3 (state transition) ve Fig. 7 (consequent step) metinde zaten anlatılıyor; iki blok diyagram birleştirilebilir. Her biri yaklaşık 0,3 sayfa.
5. **Manuscript ID satırı** başlıktan silinecek veya portalın verdiği numarayla doldurulacak.
6. **Gönderim PDF'i Word'den export edilecek** (LibreOffice denklemleri düşürüyor).
