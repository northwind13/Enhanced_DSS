# DisasterAware → IEEE Systems Journal: değişiklik listesi

Ölçüm tarihi: 2026-08. Ölçülen dosya: `02_Article/ieee_systems/DisasterAware_Article.docx` (TSMC'ye giden anonim sürüm).

## 0. Mevcut durum (ölçülmüş)

| Ölçü | Mevcut | Systems Journal hedefi |
|---|---|---|
| Sayfa (LibreOffice, Word'de genelde 1 eksik) | 10 | ≤ 12, hedef 11 |
| Toplam kelime | 7 595 | ~7 000 gövde + eklemeler |
| Figür | 13 | ≤ 10 |
| Tablo | 6 | ≤ 5 |
| Numaralı denklem | 31 | ≤ 20 |
| Kaynak | 35 | 35-45 |
| Abstract | 230 kelime | 150-250 (uygun) |
| Anahtar kelime | 5 | ≥ 5 (uygun) |
| Yazar bilgisi | YOK (anonim sürüm) | Tek körse EKLENECEK |

## 1. Formatsal değişiklikler

1. **Yazar bloğu geri gelecek.** Şablonun başlığına iki yazar adı, dipnota kurum bilgisi. Anonim sürüm TSMC çift-kör içindi; Systems Journal tek kör (portalda teyit edilecek). Maliyet: yaklaşık +0,15 sayfa.
2. **Biyografiler ve fotoğraflar eklenecek.** `ieee_systems/files/caglarakman.png` ve `kleb.jpg` mevcut. Maliyet: +0,5 ile +0,6 sayfa.
3. **Acknowledgment bölümü metne dönecek** (TeamAware / Horizon 2020 No. 101019808). Maliyet: +0,1 sayfa.
4. **Başlıktaki manuscript ID satırı** ("REPLACE THIS LINE...") gönderim öncesi temizlenecek ya da portalın verdiği numarayla doldurulacak.
5. **Gönderim PDF'i Word'den export edilecek.** LibreOffice OMML denklemlerini düşürüyor; kontrol için kullanılabilir, gönderim için kullanılamaz.
6. **Şablon aynı kalıyor:** `ieee_systems/template/template_IEEE_Systems.docx` (IEEE Transactions şablonu), çift sütun, 10 punto Times, US Letter.
7. **Denklem sayısı 31'den 20'ye inecek**; kalanlar satır içine alınacak. Kazanç: yaklaşık 0,4 sayfa.
8. **Figür 13'ten 10'a, tablo 6'dan 5'e inecek.** Düşürme adayları: Fig. 3 (state transition), Fig. 7 (consequent step), iki blok diyagramın birleştirilmesi; kriter tablosu ile uyum tablosunun tek tabloda birleşmesi. Kazanç: 0,8 ile 1,1 sayfa.
9. Şekil çözünürlüğü, caption stili ve çapraz referans alanları korunacak; kaynakça IEEE stili, alanlar İngilizce (\l 1033).

**Sayfa aritmetiği:** 10 (mevcut) + 0,85 (yazar, biyografi, teşekkür) + 2,2 (yeni içerik, aşağıda) − 2,3 (kısaltma ve budama) ≈ **10,8 sayfa**. 12 sınırının altında, ilk 8 sayfa ücretsiz olduğu için yaklaşık 3 sayfa x 150 = 450 USD ek sayfa ücreti öngörülmeli.

## 2. İçerik değişiklikleri

1. **Katkı beyanı paragrafı** (yeni, giriş bölümünün sonuna): hangi nesne tanımlanıyor, hangi özellik garanti ediliyor, ne ispatlanıyor, ne ölçülüyor. +0,15 sayfa.
2. **Dört önerme ve kısa ispatları** (yeni, Bölüm V): (i) kabul edilen hiçbir değişiklik öngörülen fiziksel maliyeti artıramaz; (ii) yaşam güvenliği emirleri hiçbir kapı tarafından azaltılamaz; (iii) kapsama koşulu sağlandıkça her erişilebilir okumada bir kural eşiğin üstünde ateşlenir, aksi halde gözetleme duruşuna düşülür; (iv) kural tabanının büyümesi sınırlıdır ve merdiven sonlanır. +0,5 ile +0,6 sayfa. Bu, editörlerin "artımlı mühendislik katkısı" itirazına verilen doğrudan cevaptır.
3. **Harici temel politika karşılaştırması** (yeni deney): aynı simülatör, aynı tohumlar, aynı senaryolar; değer ağırlıklı açgözlü tahsis politikası. Sonuç mevcut Tablo V'e sütun olarak eklenir, bir paragraf yorum yazılır. +0,3 sayfa. Bu da "harici karşılaştırma yok" itirazını kapatır.
4. **Fizik anlatımı kısalacak** (Bölüm III): yayılım, yakıt ve bastırma denklemlerinin bir kısmı satır içine alınıp anlatım sıkıştırılacak. −0,8 sayfa.
5. **Tartışma bölümü eklenecek** ("Systems Lessons and Limitations"): çıkarılan dersler, iddia sınırı (simülasyon temelli, konuşlandırma yok), insan döngüde değerlendirmenin neden sonraki adım olduğu. +0,3 sayfa. Dergi "successful lessons" vurgusu yapıyor, bu bölüm o beklentiye karşılık gelir.
6. **Sonuç bölümü kısalacak**, gelecek çalışma listesi üç maddeye inecek. −0,2 sayfa.
7. Kaynakça 35'ten 40 civarına çıkacak: sistemlerin sistemi, birlikte çalışabilirlik ve afet müdahalesi mimarileri literatüründen beş altı kaynak eklenecek (derginin kendi yayınlarından seçilecek).

## 3. Kapsam konumlandırması

Derginin kapsam dışı listesinde iki madde var ve ikisi de bizi hedef alabilir: sistemler arası bir problemi ele almayan saf bulanık sistem çalışmaları, ve tesis/denetleyici/eyleyici/algılayıcı bileşenli geri beslemeli kontrol sistemleri. Bu nedenle:

1. Sistem, **çok varlıklı bir bütün** olarak sunulacak: bağımsız veri sağlayıcılar (uydu, meteoroloji, coğrafi katmanlar, kaynak takibi), N adet yerel karar ajanı, çıkarım yapmayan koordinasyon katmanı, denetleyici insan operatör, müdahale organizasyonu.
2. Bölüm III **mimari ve arayüz sözleşmeleri** üzerine kurulacak; ayrışma değişmezi (hiçbir dış modül durumu yazamaz) bir sistem özelliği olarak ifade edilecek.
3. **Bulanık çıkarım araç düzeyine inecek**: başlıkta ve abstract'ta geçmeyecek, Bölüm V'te yöntem olarak anlatılacak.
4. **Dil modeli önerici abstract'tan çıkacak**, Bölüm VI'da kapıların arkasında değiştirilebilir bir bileşen olarak sunulacak; garantiler önericiden bağımsız ifade edilecek.
5. "plant", "controller", "actuator", "fuzzy control" ifadeleri metinden tamamen ayıklanacak.
6. İnsan-sistem etkileşimi görünür olacak: operatörün geri alma, duraklatma ve senaryo enjeksiyonu yetkisi, izlenebilirlik ve operatör güveni ayrı bir alt bölümde.
7. Cover letter, derginin kapsam metnindeki afet müdahalesi cümlesine açık atıf yapacak.

## 4. Başlık ve anahtar kelimeler

Mevcut başlık: "DisasterAware: AI Enhanced Concept-Based Fuzzy Reasoning for Scalable Wildfire Decision Support". Sorunlar: "AI Enhanced" pazarlama dili, "Fuzzy Reasoning" kapsam dışı listesini çağrıştırıyor, sistem vurgusu yok.

Önerilen başlık:
**"DisasterAware: A Distributed Decision-Support Architecture with a Verified Open Intervention Space for Wildfire Response"**

Alternatifler:
- "A System-of-Systems Architecture for Wildfire Response with a Verified Open Intervention Space"
- "Distributed, Human-Supervised Wildfire Decision Support with Runtime-Extensible Interventions"

Anahtar kelimeler (mevcut beş kelime değişecek):
system-of-systems architecture; distributed decision support; open intervention space; verification and fail-safe design; human-supervised emergency response; wildfire management

## 5. Bölüm yapısı (yeni)

```
I.    Introduction (katkı beyanı dahil)
II.   Related Work and Systems Context
III.  System Architecture and Interfaces
IV.   Observation and Concept Layers
V.    Decision Layer and Its Guarantees (Önerme 1-4)
VI.   Open Intervention Space: Staged Adaptation under Verification
VII.  Evaluation Design (harici baseline dahil)
VIII. Results
IX.   Discussion: Systems Lessons and Limitations
X.    Conclusion and Future Work
      References, Biographies
```
