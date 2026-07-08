# Harita Gerçekçilik Kuralları (procedural generation)

Bu belge, `disaster_phyengine/terrain.py` içindeki `generate_landscape`
fonksiyonunun uyduğu (ve uyması gereken) kuralları toplar. [x] = uygulanmış.

## 1. Su (deniz ve göl)
- [x] Deniz, arazi kota göre denize doğru alçaldıktan sonra, deniz seviyesinin
  altındaki hücrelerin harita KENARINDAN "flood-fill" ile doldurulmasıyla
  oluşur. Böylece kıyı araziyi izler: körfezler, yarımadalar ve denizde kalan
  yüksek noktalar ADA olur (düz falez değil).
- [x] Kıyıda falez yok: kara, kıyıya doğru bir bant boyunca kotu sıfıra
  rampalar (plaj eğimi).
- [x] Göller = deniz seviyesi altındaki, kenara BAĞLI OLMAYAN iç çukurlar
  (ayrı su kütleleri).
- [x] Su hücreleri yanmaz (non-fuel).

## 2. Nehir
- [x] Tek, sürekli (kesintisiz) bir su yolu; hücre hücre izlendiği için
  kopuk/dashed değil.
- [x] Yokuş-aşağı akar: en yüksek kaynaktan başlar, her adımda daha alçak
  komşuya gider; ASLA yukarı çıkmaz.
- [x] Denize, bir göle veya harita kenarına dökülür (ağız). Kapalı çukurda
  biterse orada durur (küçük bir baseni besler).
- [x] Araziye göre doğal kıvrım (meander) yapar; düz çizgi değil.

## 3. Nehir–Yol ilişkisi
- [x] Yol nehri KÖPRÜYLE geçebilir: su hücresinde roads/access işaretlenir
  ama hücre su kalır (kurumaz, yanmaz). Maliyet yüksek (18/hücre) olduğu
  için köprü mümkün olan en kısa, dike yakın geçiştir.
- [x] Yol suyun ORTASINDAN/boyunca gitmez; nehir yanından (banktan) geçebilir
  ama aynı hücre şeridinde değil. Göl/deniz üstünden uzun geçiş fiilen
  yasak (maliyet).

## 4. Yollar
- [x] En-düşük-maliyet yol bulma (Dijkstra) ile ağaç yapısında ağ:
  kasabadan kök alan tek en-kısa-yol ağacı, tüm yerleşimleri bağlar.
- [x] Maliyet eğimle artar → yollar düşük kotu/vadileri izler; dik yamaçta
  düz tırmanmak yerine dolaşarak (switchback) çıkar.
- [x] Yol HER ZAMAN haritayı bir kenardan terk eder (harita-dışı bağlantı).
- [x] Şehir içinde sokak ızgarası (bloklar arası caddeler).

## 5. Yol–Su (deniz/göl) ilişkisi
- [x] Yol suya (deniz/göl/nehir) ASLA girmez; su hücreleri geçilmez.
- [x] Yol gölü/denizi ortadan kesmez; etrafından dolaşır, kıyıda durur.
- [x] Son adımda güvenlik: `roads &= (ftype != WATER)` — suda yol hücresi kalmaz.

## 6. Şehirler / Yerleşimler
### 6.1 Konum
- [x] Uygun karaya konur: düşük kot, az eğim, suya yakın ama üstünde değil.
### 6.2 Dağılım
- [x] 2B mavi-gürültü (blue-noise) ile haritaya serpiştirilir; aralarında
  minimum mesafe var → tek bir yola/hatta dizilmezler, kümelenmezler.
- [x] Her seed farklı düzen üretir.
### 6.3 Şekil / boyut
- [x] Küçük daire DEĞİL: yapılı alan nüfusla büyür. Büyük nüfus = GENİŞ şehir,
  içinde sokak ızgarası olan blok deseni (built-in `city_wui` mantığı).
- [x] Köyler küçük yapılı alan, kasaba/şehir geniş ızgara.
### 6.4 İçerik (çeşitli bileşenler)
- [x] Sadece "2 bina" değil: Hospital, Power plant, Water treatment,
  Government office, School, Fire station, Telecom tower, Fuel depot +
  konut/şehir merkezi + nüfus.
- [x] Büyük şehir tüm tesis setini alır; köyler nüfusla ölçekli rastgele
  bir alt küme, HER köyde en az 1 tesis (okul/itfaiye ölçeği).
- [x] Yerleşim sayısı 50'ye kadar seçilebilir (UI slider).
- [x] Nüfus girdisi haritanın TOPLAM nüfusudur; çarpık paylarla dağıtılır
  (kasaba en az %35, köyler küçülen paylar) ve paylar tam toplama eşittir.
- [x] Evacuation route, denizde değil; yolun haritadan çıktığı kenar noktada.

## 7. Yapılar (genel)
- [x] HİÇBİR yapı (urban blok, bina, tesis, nüfus, evac) su üstüne konmaz;
  yerleştirme en yakın kara hücresine kaydırılır (`_nearest_land`).

## 8. Meteoroloji (global değil, lokal)
- [x] Rüzgar uzamsal alan: taban akış + fraktal gustiness; açık/yüksek
  zeminde güçlü, korunaklı vadide zayıf; yön yer yer sapar.
- [x] Nem uzamsal: kot + su yakınlığı (riparian) + fraktal + yağış.
- [x] Yağış (precipitation) uzamsal: dağınık sağanaklar; yakıtı ıslatıp
  nemi artırır → yağışlı hücreler yangına direnir.
- [x] Sıcaklık ve bağıl nem de haritada değişir.

## 9. Rölyef ve bitki örtüsü gerçekçiliği (erozyon + bakı + zonlama)
- [x] Erozyon geçişi: D8 akış birikimi üzerinden stream-power oyma
  (`_flow_accumulation` + `_erode`); fBm'nin izole çukurları BAĞLANTILI,
  dendritik vadilere dönüşür. Drenaj haritası nem için de kullanılır.
- [x] Bakı-bağımlı bitki örtüsü: güneye bakan yamaçlar kuru ot/çalı,
  kuzeye bakanlar yoğun orman (güneş indeksi eğimle ağırlıklı).
- [x] Yükseklik zonlaması: vadi tabanı yapraklı, orta kot kızılçam kuşağı,
  sırt çalı; sırt yakınında yakıt yükü inceler.
- [x] Riparian şerit: su kenarındaki kara hücreleri nemli yapraklı örtü.
- [x] Kaya çıkıntıları: dik + yüksek hücrelerde çıplak kaya (doğal yakıt
  kesintisi, ftype=0).
- [x] Yakıt yamalılığı: eski yanık izleri (0-2 düzensiz leke, düşük yük ot)
  ve yerleşim çevresinde dikdörtgen tarım parselleri.
- [x] Nem gradyanı: topografik ıslaklık (drenaj**0.7) + su yakınlığı;
  vadi tabanı sırttan belirgin nemli başlar.
- [x] Yerleşim skoru suya yakınlığı da tartar (üstüne değil, yanına).
- [x] Su kütleleri bütündür: 4 hücreden küçük su kırıntıları temizlenir
  (`_prune_puddles`, 8-komşuluk; çapraz akan nehir tek gövde sayılır).
- [x] Yol su hücresini ASLA ezmez: `add_road_disk/rect` su ftype'ını
  değiştirmez (eskiden nehir kıyısını kemiriyordu).

## Açık konular / iyileştirme adayları
- [ ] vpop yoğunluğu küçük hücrede çok yüksek görünüyor (kişi/km²); cost
  içsel tutarlı ama gösterim için gerçekçi yoğunluk tavanı eklenebilir.
- [ ] relief_m küçük haritada aşırı dik eğim üretebilir; relief'i harita
  fiziksel boyutuna göre ölçekleme opsiyonu.
- [ ] Yağışın çalışma sırasında dinamik etkisi (yağmur adım adım nem ekler)
  şu an yok; başlangıç nemine kuruluyor.
