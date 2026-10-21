const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require("/root/.claude/skills/synced/4254a022-cd98-4a81-8e81-565ff344659a_a1fc28ed-95ec-445b-a864-79baa487085f/pptx/scripts/apply_theme.js");

// ---------------------------------------------------------------- theme
const THEME = {
  name: "TOK Akademik",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1C2430",   // near-black text
    lt1: "FFFFFF",
    dk2: "1F3A5F",   // deep navy (dominant)
    lt2: "EEF1F5",   // cool light grey
    accent1: "8B2635", // burgundy
    accent2: "3C6E9F", // steel blue
    accent3: "C9A227", // muted gold
    accent4: "6B7A8F", // slate grey
    accent5: "D7DEE8", // pale blue grey
    accent6: "2E7D5B", // muted green
    hlink: "3C6E9F",
    folHlink: "6B7A8F",
  },
};
const H = THEME.colors;

async function icon(name, color, px = 256) {
  const el = React.createElement(fa[name], { color: "#" + color, size: px });
  const svg = ReactDOMServer.renderToStaticMarkup(el);
  const buf = await sharp(Buffer.from(svg)).resize(px, px).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  pres.author = "Çağlar Akman";
  pres.title = "Kural tabanlı karar destek sistemlerinde açık karar uzayı";
  const C = pres.SchemeColor;
  const W = 13.33, M = 0.6;

  // ------------------------------------------------------------ layouts
  const footerCommon = (dark) => ({
    text: "Akman, Leblebicioğlu | TOK 2026 | ODTÜ EEE",
    options: { x: M, y: 7.0, w: 8, h: 0.35, fontSize: 12, color: dark ? C.accent5 : C.accent4, margin: 0 },
  });
  const slideNum = (dark) => ({ x: W - M - 0.8, y: 7.0, w: 0.8, h: 0.35, fontSize: 12, align: "right", color: dark ? C.accent5 : C.accent4, margin: 0 });

  pres.defineSlideMaster({
    title: "TITLE_DARK",
    background: { color: H.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 2.1, w: W - 2 * M, h: 2.0, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "Başlık" } },
      { placeholder: { options: { name: "body", type: "body", x: M, y: 4.25, w: W - 2 * M, h: 1.0, fontSize: 22, color: C.accent5, valign: "top", margin: 0 }, text: "Alt başlık" } },
      { placeholder: { options: { name: "authors", type: "body", x: M, y: 5.6, w: W - 2 * M, h: 1.2, fontSize: 18, color: C.background1, valign: "top", margin: 0 }, text: "Yazarlar" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT",
    background: { color: H.lt1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.9, fontSize: 34, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "Başlık" } },
      { text: footerCommon(false) },
    ],
    slideNumber: slideNum(false),
  });
  pres.defineSlideMaster({
    title: "CONTENT_DARK",
    background: { color: H.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 0.9, fontSize: 34, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "Başlık" } },
      { text: footerCommon(true) },
    ],
    slideNumber: slideNum(true),
  });

  // helpers
  const card = (s, x, y, w, h, opts = {}) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.12, fill: { color: opts.fill || C.background2 }, line: { color: opts.line || C.accent5, width: 0.75 },
    shadow: { type: "outer", color: H.accent4, blur: 4, offset: 1, angle: 90, opacity: 0.18 }, objectName: opts.name || "card",
  });
  const circleIcon = async (s, name, x, y, d, bg, fg) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: bg }, line: { color: bg, width: 0 }, objectName: name + " circle" });
    s.addImage({ data: await icon(name, fg), x: x + d * 0.25, y: y + d * 0.25, w: d * 0.5, h: d * 0.5, objectName: name + " icon" });
  };
  const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 20, valign: "top" }, o));

  // ============================================================ 1 Title
  pres.addSection({ title: "Giriş" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Giriş" });
  s.addText("Kural tabanlı karar destek sistemlerinde açık karar uzayı", { placeholder: "title" });
  s.addText("Büyük dil modeli destekli kural üretimi ve belirlenimci doğrulama", { placeholder: "body" });
  s.addText([
    { text: "Çağlar Akman, Mehmet Kemal Leblebicioğlu", options: { bold: true, breakLine: true } },
    { text: "Elektrik ve Elektronik Mühendisliği Bölümü, Orta Doğu Teknik Üniversitesi", options: { fontSize: 16, color: C.accent5 } },
  ], { placeholder: "authors" });
  s.addShape(pres.shapes.OVAL, { x: W - M - 1.6, y: 0.7, w: 1.3, h: 1.3, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "mark" });
  s.addImage({ data: await icon("FaProjectDiagram", H.lt1), x: W - M - 1.28, y: 1.02, w: 0.66, h: 0.66, objectName: "mark icon" });
  s.addNotes("0:00-0:30. Açılış. Bildirinin konusu: kural tabanlı karar destek sistemlerinin kapalı karar uzayı sorunu ve bunu çalışma sırasında, denetim altında açan üç aşamalı bir uyarlama mimarisi. Örnek uygulama orman yangını müdahalesi.");

  // ============================================================ 2 Problem
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Giriş" });
  s.addText("Sorun: karar uzayı tasarımda kapanır", { placeholder: "title" });
  const rows = [
    ["FaBookOpen", "Kurallar tasarım aşamasında yazılır", "EĞER-İSE kuralları izlenebilir, ancak çalışma sırasında değişmez."],
    ["FaExclamationTriangle", "Öngörülmemiş durumda yanıt yok", "Sensör kaybı, beklenmedik dinamik, görülmemiş durum bileşimi: ya hiçbir kural ateşlenmez ya da yetersiz kalır."],
    ["FaEyeSlash", "Kara kutu politikalar gerekçe vermez", "Katılığı giderir, ancak kritik alanlarda güven ve hesap verebilirlik kaybolur."],
  ];
  let y = 1.75;
  for (const [ic, h1, h2] of rows) {
    await circleIcon(s, ic, M, y + 0.05, 0.9, H.dk2, H.lt1);
    txt(s, h1, { x: M + 1.2, y, w: 6.4, h: 0.5, fontSize: 24, bold: true, color: C.text2 });
    txt(s, h2, { x: M + 1.2, y: y + 0.52, w: 6.1, h: 0.9, fontSize: 18, color: C.text1 });
    y += 1.6;
  }
  card(s, 8.3, 1.75, 4.45, 4.7, { fill: C.background2, name: "question card" });
  txt(s, "Soru", { x: 8.65, y: 2.05, w: 3.8, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  txt(s, "Karar uzayı çalışma sırasında, izlenebilirlik korunarak nasıl açık tutulur?", { x: 8.65, y: 2.6, w: 3.8, h: 2.2, fontSize: 24, color: C.text2, bold: true });
  txt(s, "Yanıt: yeni bilgi bir dil modelinden gelir, kabul kararı yalnızca belirlenimci doğrulamaya dayanır.", { x: 8.65, y: 4.75, w: 3.8, h: 1.5, fontSize: 18, color: C.text1 });
  s.addNotes("0:30-1:30. Kural tabanlı KDS'nin gücü izlenebilirlik, zayıflığı kapalılık. Gerçek operasyonda öngörülmemiş durum sık. Öğrenilmiş politikalar esnek ama gerekçesiz. Bildirinin sorusu ve ilkesi: bilgi modelden, kabul kararı kapılardan.");

  // ============================================================ 3 Architecture


  // ============================================================ 2a Glossary
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Giriş" });
  s.addText("Terimler", { placeholder: "title" });
  const glh = { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: 14, valign: "middle" };
  const glc = { fontSize: 13, color: H.dk1, valign: "middle" };
  const glt = { fontSize: 13, color: H.dk2, bold: true, valign: "middle" };
  const terms = [
    ["Karar uzayı", "decision space", "Sistemin bir anda verebileceği emirlerin kümesi ve bu kümenin genişletilme kuralları"],
    ["Kapalı / açık karar uzayı", "closed / open", "Tasarımda sabitlenen küme; çalışma sırasında doğrulanarak büyüyen küme"],
    ["Tohum kural tabanı (F5)", "seed rule base", "Beş EĞER-İSE kuralından oluşan başlangıç doktrini"],
    ["Gelişen bulanık sistem (evFIS)", "evolving fuzzy system", "Aşama 1-2: kural parametrelerini ve terim çözünürlüğünü çevrimiçi ayarlayan katman"],
    ["Üretken öneri", "generative proposal", "Aşama 3: dil modelinin şema içinde önerdiği kural, kavram veya müdahale paketi"],
    ["Makro / eyleyici / ara kavram", "macro / actuator / concept", "Mevcut müdahale demeti / doğrulanmış etkilerden yeni müdahale / kavramların doğrusal bileşimi"],
    ["Koşul (clause)", "clause", "Etki + sektör + hücre bandı + miktar; konumlandırılmış emrin en küçük birimi"],
    ["Gölge benzetim", "shadow forecast", "Adayı icra etmeden iki bağımsız tohumla ileriye koşturan tahmin"],
    ["Doğrulama kapısı (G1-G5b)", "verification gate", "Belirlenimci kabul koşulu; kabul kararı önericiye bırakılmaz"],
    ["Yeterlilik eşiği", "satisficing bound", "Öngörülen maliyet bu eşiğin üstündeyse uyarlama tetiklenir"],
    ["Eşleştirilmiş dünya", "paired world", "Aynı tohum, aynı arazi ve tutuşma; her yapılandırma aynı dünyayı oynatır"],
    ["Karar maliyeti J", "decision cost", "Yanma, varlık, nüfus (fiziksel) + müdahale, gecikme (yanıt) terimlerinin ağırlıklı toplamı"],
  ];
  const grows = [[{ text: "Terim", options: glh }, { text: "İngilizce", options: glh }, { text: "Anlamı", options: glh }]]
    .concat(terms.map(([a, b, c]) => [{ text: a, options: glt }, { text: b, options: Object.assign({}, glc, { italic: true, color: H.accent4 }) }, { text: c, options: glc }]));
  s.addTable(grows, { x: M, y: 1.45, w: 12.13, colW: [3.0, 2.3, 6.83], rowH: 0.4, border: { type: "solid", color: H.accent5, pt: 0.75 }, fontFace: "+mn-lt", objectName: "glossary table" });
  s.addNotes("Referans slaydı; sunumda hızlı geçilir (15 s) veya soru-cevapta açılır. Türkçe karşılıklar bildirideki terimlerle aynı.");

  pres.addSection({ title: "Mimari" });
  // ============================================================ 2b Vocabulary of the decision layer
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mimari" });
  s.addText("Karar katmanı: özellik → kavram → kural → müdahale", { placeholder: "title" });
  const cols = [
    ["10 gözlem özelliği", "her biri güven değeri taşır", ["Yangın şiddeti", "Yayılma potansiyeli", "Hava şiddeti", "Tutuşma yakınlığı", "Yakıt yükü", "Varlık maruziyeti", "Kaynak erişilebilirliği", "Yol ve erişim durumu", "Söndürme kapasitesi", "Zamansal aciliyet"], 14],
    ["5 karar kavramı", "kurallar yalnızca bunlar üzerine yazılır", ["Yangın tehdit düzeyi", "Söndürme uygulanabilirliği", "Varlık maruziyet riski", "Müdahale aciliyeti", "Tahliye baskısı"], 17],
    ["6 müdahale ailesi", "her kural bir şiddet atar; 0 = eylemsizlik", ["Söndürme çabası", "Kaynak konuşlandırma", "Kontrol hattı", "Varlık koruma", "Tahliye", "Kamu uyarısı"], 17],
  ];
  let vx = M;
  for (const [h1, h2, items, fs] of cols) {
    card(s, vx, 1.5, 3.85, 4.75, { name: "vocab " + h1 });
    txt(s, h1, { x: vx + 0.25, y: 1.65, w: 3.4, h: 0.45, fontSize: 21, bold: true, color: C.accent1 });
    txt(s, h2, { x: vx + 0.25, y: 2.1, w: 3.4, h: 0.4, fontSize: 14, color: C.accent4, italic: true });
    txt(s, items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })), { x: vx + 0.25, y: 2.6, w: 3.45, h: 3.55, fontSize: fs, paraSpaceAfter: fs > 15 ? 7 : 3 });
    // arrow between columns
    if (vx + 3.85 < 12) s.addShape(pres.shapes.RIGHT_ARROW, { x: vx + 3.9, y: 3.55, w: 0.35, h: 0.5, fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 }, objectName: "arrow" });
    vx += 4.3;
  }
  card(s, M, 6.35, 12.13, 0.55, { fill: C.accent5, line: C.accent5, name: "physics card" });
  txt(s, "Altta benzetimin doğruladığı 7 fiziksel etki (ıslatma, temizleme, tutuşturma, kaplama, su çekme, tahliye, hazırlama) ve 3 kanal (su ikmali, geciktirici atışı, taktik yakma). Tohum tabanı: 5 kural; öncül uzayı 5⁵ = 3125 bileşim, yalnızca etkin alt küme kurulur.", { x: M + 0.2, y: 6.35, w: 11.8, h: 0.55, fontSize: 13, color: C.text2, valign: "middle" });
  s.addNotes("1:15-1:45. Karar katmanının sözlüğü. On gözlem özelliği, her biri bir güven değeriyle gelir; güven düşükse kavram kalıcılık tahminiyle harmanlanır. Beş karar kavramı komutanın sorduğu sorulardır; kurallar yalnızca bu beş kavram üzerine yazıldığı için kural yükü çarpımdan toplama iner. Altı müdahale ailesi her kuralı icra edilebilir bir emre bağlar; eylemsizlik her ailenin sıfır noktasıdır, maliyet eylemsizliği de fiyatlar. Aile listesi kapalıdır; açıklık yeni kurallar ve aynı aileler üzerinde bileşimlerle gelir. Bir alt katmanda yedi fiziksel etki ve üç kanal vardır; üçüncü aşamanın önerdiği yeni müdahaleler bu etkilerden kurulur.");

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mimari" });
  s.addText("Üç aşamalı uyarlama döngüsü", { placeholder: "title" });
  s.addImage({ path: "fig1.png", x: M, y: 1.45, w: 9.3, h: 5.26, objectName: "Şekil 1" });
  const steps = [
    ["1", "Parametre uyarlaması", "Mevcut kuralların şiddetleri ve terim sınırları ayarlanır."],
    ["2", "Çözünürlük büyütme", "Boş öncül bileşimlerine kural yazılır, terim kümesi genişler."],
    ["3", "Üretken öneri", "Dil modeli yeni kural, kavram ve bileşik müdahale önerir."],
  ];
  y = 1.45;
  for (const [n, h1, h2] of steps) {
    s.addShape(pres.shapes.OVAL, { x: 10.15, y: y, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1, width: 0 }, objectName: "step " + n });
    txt(s, n, { x: 10.15, y: y, w: 0.5, h: 0.5, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, h1, { x: 10.15, y: y + 0.55, w: 2.6, h: 0.35, fontSize: 18, bold: true, color: C.text2 });
    txt(s, h2, { x: 10.15, y: y + 0.9, w: 2.6, h: 0.55, fontSize: 15 });
    y += 1.5;
  }
  card(s, 10.15, 6.05, 2.6, 0.65, { fill: C.accent5, line: C.accent5, name: "rule card" });
  txt(s, "Aday: ΔĴphys < 0 ise tutulur", { x: 10.25, y: 6.05, w: 2.45, h: 0.65, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
  s.addNotes("1:30-2:45. Şekil 1. Değer tabanlı seçici (bağlamsal haydut) bir aşama çağırır; aday değişiklik doğrulama kapılarından geçer; yalnızca kabul edilirse kural tabanına işlenir. Uyarlama iki belirti üzerine tetiklenir: maliyet açığı (kural ateşleniyor ama yetersiz) ve kapsama boşluğu (hiçbir kural ateşlenmiyor). Aşama 1 ve 2 gelişen bulanık sistem geleneği; aşama 3 sözlüğün kendisini genişletir. Seçici: alfa 0,05, epsilon 0,10; ödül öngörülen maliyet düşüşü.");

  // ============================================================ 4 Stage 3
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mimari" });
  s.addText("Aşama 3: şema kısıtlı üretken öneri", { placeholder: "title" });
  // left: inputs
  txt(s, "Modele verilen", { x: M, y: 1.6, w: 3.8, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  const ins = [["FaUserCog", "Rol tanımlayan sistem talimatı"], ["FaClipboardList", "Çevrim durumu: kavramlar, etkin kurallar, kapsama, kaynak"], ["FaLock", "Şema kısıtı: yalnızca sistemin kapalı sözlüğü"]];
  y = 2.2;
  for (const [ic, t] of ins) {
    await circleIcon(s, ic, M, y, 0.7, H.dk2, H.lt1);
    txt(s, t, { x: M + 0.9, y: y, w: 2.9, h: 1.0, fontSize: 17, valign: "middle" });
    y += 1.2;
  }
  // arrow
  s.addShape(pres.shapes.RIGHT_ARROW, { x: 4.75, y: 3.2, w: 0.8, h: 0.7, fill: { color: C.accent3 }, line: { color: C.accent3, width: 0 }, objectName: "arrow" });
  // middle: output
  card(s, 5.75, 1.6, 3.3, 4.6, { name: "output card" });
  txt(s, "Tek öneri", { x: 6.0, y: 1.85, w: 2.9, h: 0.5, fontSize: 20, bold: true, color: C.accent1 });
  txt(s, "Bir kural, ya da bir kural + tek bir yeni sözlük nesnesi:", { x: 6.0, y: 2.4, w: 2.9, h: 0.9, fontSize: 17 });
  const outs = [["Makro müdahale", "en çok 3 kanal, sabit oranlar"], ["Ara kavram", "en çok 4 girdinin normalize bileşimi"], ["Koşul eyleyicisi", "doğrulanmış etki + sektör + hücre bandı"]];
  y = 3.35;
  for (const [a, b] of outs) {
    s.addShape(pres.shapes.OVAL, { x: 6.0, y: y + 0.12, w: 0.18, h: 0.18, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 }, objectName: "dot" });
    txt(s, [{ text: a, options: { bold: true, color: C.text2, breakLine: true } }, { text: b, options: { fontSize: 15, color: C.accent4 } }], { x: 6.3, y: y, w: 2.65, h: 0.9, fontSize: 17 });
    y += 0.92;
  }
  // right: principle
  card(s, 9.4, 1.6, 3.35, 4.6, { fill: H.dk2, line: H.dk2, name: "principle card" });
  txt(s, "İlke", { x: 9.7, y: 1.85, w: 2.8, h: 0.5, fontSize: 20, bold: true, color: C.accent3 });
  txt(s, "Müdahale düzeyinde açık, fiziksel etki düzeyinde kapalı", { x: 9.7, y: 2.4, w: 2.8, h: 1.6, fontSize: 24, bold: true, color: C.background1 });
  txt(s, "Model bağlam içinde kullanılır; ağırlık güncellemesi ve göreve özel eğitim yoktur. Reddedilen öneri, kapı adıyla birlikte sınırlı düzeltme hakkıyla geri döner.", { x: 9.7, y: 4.1, w: 2.8, h: 2.0, fontSize: 16, color: C.accent5 });
  s.addNotes("2:45-3:45. Önerici önceden eğitilmiş genel amaçlı bir LLM. Üç girdi, tek çıktı. Yeni nesne üç türden biri: makro, ara kavram, koşul eyleyicisi. Koşul eyleyicisi yeni müdahale tanımlar ama çağırabileceği fiziksel etkiler benzetimde doğrulanmış sabit kümeyle sınırlı. Günce: önceki kabul/retler isteme eklenir, yinelenen öneri önlenir; kabul edilen nesneler soy kütüğüne yazılır.");

  // ============================================================ 5 Gates
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Mimari" });
  s.addText("Doğrulama kapıları: kabul kararı modele bırakılmaz", { placeholder: "title" });
  const gh = { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: 15, valign: "middle" };
  const gc = { fontSize: 15, color: H.dk1, valign: "middle" };
  const gcb = Object.assign({}, gc, { bold: true, color: H.dk2 });
  const gateRows = [
    [{ text: "Kapı", options: gh }, { text: "Denetlenen koşul", options: gh }, { text: "Tür", options: gh }],
    [{ text: "G1 Biçim", options: gcb }, { text: "Öncül ve sonuç iyi biçimli bir kural oluşturur", options: gc }, { text: "cebirsel", options: gc }],
    [{ text: "G2 Kısıt", options: gcb }, { text: "Sözlük, terim sayısı ve aralıklar geçerli; dolu bileşim yinelenmez", options: gc }, { text: "cebirsel", options: gc }],
    [{ text: "G2b Artıksızlık*", options: gcb }, { text: "Yeni kavram mevcut bir kavramı yinelemez (cos < 0,95)", options: gc }, { text: "cebirsel", options: gc }],
    [{ text: "G2c İlgililik", options: gcb }, { text: "Önerinin tüm öncülleri mevcut durumda ateşlenir", options: gc }, { text: "cebirsel", options: gc }],
    [{ text: "G2d Havuz ekonomisi", options: gcb }, { text: "Bütçe doluyken fiziksel iş ekleyen öneri reddedilir", options: gc }, { text: "cebirsel", options: gc }],
    [{ text: "G3 İlk kazanç", options: gcb }, { text: "Öngörülen fiziksel maliyet düşer: ΔĴphys < 0", options: gc }, { text: "benzetim (a)", options: gc }],
    [{ text: "G4 Gürbüzlük", options: gcb }, { text: "Aynı denetim bağımsız ikinci tohumda da geçer", options: gc }, { text: "benzetim (b)", options: gc }],
    [{ text: "G5 Sözlük getirisi*", options: gcb }, { text: "Kazanç iki koşuda da δv marjını aşar", options: gc }, { text: "benzetim", options: gc }],
    [{ text: "G5b Katkı ayrımı*", options: gcb }, { text: "Yeni nesne çıkarıldığında maliyet iki koşuda da yükselir", options: gc }, { text: "benzetim", options: gc }],
  ];
  s.addTable(gateRows, { x: M, y: 1.55, w: 9.1, colW: [2.2, 5.3, 1.6], rowH: 0.42, border: { type: "solid", color: H.accent5, pt: 0.75 }, fill: { color: H.lt1 }, fontFace: "+mn-lt", objectName: "gates table" });
  txt(s, "* yalnızca paket önerilere uygulanır", { x: M, y: 6.45, w: 8, h: 0.4, fontSize: 13, color: C.accent4, italic: true });
  card(s, 10.0, 1.55, 2.75, 4.75, { fill: C.background2, name: "gate note" });
  await circleIcon(s, "FaShieldAlt", 10.25, 1.85, 0.8, H.accent1, H.lt1);
  txt(s, "İki bağımsız gelecek", { x: 10.25, y: 2.85, w: 2.3, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
  txt(s, "Tek koşudaki iyileşme rastlantı olabilir. İki tohumda korunmayan kazanç kabul edilmez.", { x: 10.25, y: 3.4, w: 2.3, h: 1.5, fontSize: 15 });
  txt(s, "Sözlük yalnızca gerçekten katkı sağlayan nesnelerle büyür (G5, G5b).", { x: 10.25, y: 4.95, w: 2.3, h: 1.3, fontSize: 15, bold: true, color: C.accent1 });
  s.addNotes("3:45-4:45. İlk dört kapı cebirsel, benzetim maliyeti yok; ret kapı adıyla modele döner. G3 ve G4 gölge benzetimde iki bağımsız tohum. Paketlere ek: G5 marj, G5b katkı ayrımı (yeni nesnenin emri çıkarılınca maliyet yükselmeli, yoksa kazanç kuraldaki olağan emirlere aittir). Önerici deterministik değil, ama kabul kararları modele dayanmadığı için süreç denetlenebilir.");

  // ============================================================ 6 Evaluation setup
  pres.addSection({ title: "Değerlendirme" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Değerlendirme" });
  s.addText("Kullanım örneği: orman yangını müdahalesi", { placeholder: "title" });
  const stats = [["5", "karar kavramı"], ["6", "müdahale ailesi"], ["5", "senaryo ailesi"], ["30", "eşleştirilmiş dünya / senaryo"]];
  let x = M;
  for (const [n, l] of stats) {
    card(s, x, 1.6, 2.85, 1.7, { name: "stat " + l });
    txt(s, n, { x, y: 1.65, w: 2.85, h: 1.0, fontSize: 48, bold: true, color: C.accent1, align: "center", valign: "middle" });
    txt(s, l, { x, y: 2.65, w: 2.85, h: 0.5, fontSize: 17, color: C.accent4, align: "center" });
    x += 3.09;
  }
  // scenario table
  const sh = { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: 16, valign: "middle" };
  const sc = { fontSize: 16, color: H.dk1, valign: "middle" };
  const scen = [
    [{ text: "Senaryo", options: sh }, { text: "Arazi", options: sh }, { text: "Kaynak", options: sh }, { text: "Tutuşma", options: sh }],
    ["S1", "Uzak orman", "Yeterli", "1, uzak"],
    ["S2", "Uzak orman", "Kısıtlı", "2, uzak"],
    ["S3", "Kent-orman arayüzü (12 yerleşim)", "Yeterli", "1, arayüzde"],
    ["S4", "Kent-orman arayüzü", "Kısıtlı", "2, arayüzde"],
    ["S5", "S4, gözlem kesintisi ve gürültü", "Kısıtlı", "2, arayüzde"],
  ].map((r, i) => i === 0 ? r : r.map((t, j) => ({ text: t, options: j === 0 ? Object.assign({}, sc, { bold: true, color: H.dk2 }) : sc })));
  s.addTable(scen, { x: M, y: 3.65, w: 7.3, colW: [1.1, 3.6, 1.2, 1.4], rowH: 0.42, border: { type: "solid", color: H.accent5, pt: 0.75 }, fontFace: "+mn-lt", objectName: "scenario table" });
  // configurations
  card(s, 8.3, 3.65, 4.45, 2.6, { fill: C.background2, name: "configs" });
  txt(s, "Karşılaştırılan yapılandırmalar", { x: 8.6, y: 3.85, w: 3.9, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  txt(s, [
    { text: "KDS yok: serbest yanma", options: { bullet: true, breakLine: true } },
    { text: "F5: beş tohum kuralı, statik", options: { bullet: true, breakLine: true } },
    { text: "DisasterAware: üç aşama, LLM önerici, açık karar uzayı", options: { bullet: true, bold: true, color: C.accent1 } },
  ], { x: 8.6, y: 4.35, w: 3.9, h: 1.8, fontSize: 17, paraSpaceAfter: 6 });
  txt(s, "72×50 hücre, 30 m; fiziksel adım 2 dk, karar çevrimi 12 dk, ufuk 24 dk; senaryo başına aynı 30 dünya, toplam 600 koşu; olay söndürmeye veya 12 saate kadar. Ölçütler: yanan alan, etkilenen nüfus, olay süresinde söndürme; Jtotal = (Jyanma + Jvarlık + Jnüfus + 0,2 Jmüdahale + 0,2 Jgecikme) / 3,4", { x: M, y: 6.3, w: 12.1, h: 0.6, fontSize: 13, color: C.accent4, italic: true });
  s.addNotes("4:45-5:30. Bölgesel ajanlı bulanık KDS; ajan beş kavram üzerinde çıkarım yapar, altı müdahale ailesine şiddet atar. Eşleştirilmiş Monte Carlo karşılaştırması: 5 senaryo ailesi, senaryo başına aynı 30 dünya, toplam 600 koşu. Aşama 3 bu kampanyada doğrudan büyük dil modeli ile beslendi (çağrı başına bir öneri, bir düzeltme hakkı); her öneri aynı kapı dizisinden geçti. Bütün yapılandırmalar aynı dünyayı, aynı tutuşmaları ve aynı rastgele akışı oynatır; fark yalnızca karar mekanizmasından. Not: bu sayılar bildirideki Tablo 3'ten değil, kapı dizisi tamamlandıktan sonra yapılan güncel kampanyadandır; soru gelirse böyle söyle.");

  // ============================================================ 7 Adaptation activity (native chart)
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Değerlendirme" });
  s.addText("Uyarlama seçici çalışır: 788 öneriden 338 kabul", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Deneme", labels: ["Aşama 1 (parametre)", "Aşama 2 (çözünürlük)", "Aşama 3 (üretken)"], values: [2093, 1411, 788] },
    { name: "Kabul", labels: ["Aşama 1 (parametre)", "Aşama 2 (çözünürlük)", "Aşama 3 (üretken)"], values: [451, 469, 338] },
  ], {
    x: M, y: 1.55, w: 7.4, h: 4.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [H.accent5, H.dk2], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 14, dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 14, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 12, valAxisLabelColor: H.accent4, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: H.accent5, size: 0.5 }, catGridLine: { style: "none" }, valAxisMaxVal: 2500, valAxisMinVal: 0, valAxisMajorUnit: 500,
    showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: H.dk1,
    showTitle: false, objectName: "adaptation chart",
  });
  const prods = [["22%", "kabul, aşama 1"], ["33%", "kabul, aşama 2"], ["43%", "kabul, aşama 3"]];
  y = 1.6;
  x = 8.5;
  for (const [n, l] of prods) {
    txt(s, n, { x, y, w: 1.3, h: 0.6, fontSize: 28, bold: true, color: C.accent1, valign: "middle" });
    txt(s, l, { x: x + 1.4, y, w: 2.8, h: 0.6, fontSize: 17, color: C.text1, valign: "middle" });
    y += 0.68;
  }
  card(s, 8.5, 3.75, 4.25, 2.7, { fill: C.background2, name: "products card" });
  txt(s, "Üretken aşamanın ürünleri", { x: 8.8, y: 3.95, w: 3.7, h: 0.45, fontSize: 18, bold: true, color: C.text2 });
  txt(s, [
    { text: "338 kabul: 67 yalnız kural, 271 paket", options: { bullet: true, breakLine: true } },
    { text: "185 eyleyici paketi (doğrulanmış etkilerden yeni müdahale), 57 makro, 29 ara kavram", options: { bullet: true, breakLine: true } },
    { text: "Ret: 276 paket marjı (G5), 155 iyileşme yok (G3), 12 çift tohum (G4)", options: { bullet: true } },
  ], { x: 8.8, y: 4.45, w: 3.75, h: 1.95, fontSize: 16, paraSpaceAfter: 4 });
  s.addNotes("5:30-6:15. 150 dünya, 788 LLM önerisi, 338 kabul (yüzde 43). Düşük kabul oranı tasarımın beklenen sonucu: adaylar gölge benzetimde ucuz sınanır ve güvenle reddedilir; retlerin çoğu tahmin testlerini geçtikten sonra paket marjında (G5) düşer. Kabul türe göre değişir: mevcut müdahaleleri paketleyen makrolar yüzde 80, doğrulanmış etkilerden yeni müdahale kuran eyleyici paketleri yüzde 46, ara kavramlar yüzde 37, mevcut sözlük içinde yalnız kurallar yüzde 28. Sözlüğü genişleten nesneler, sözlük içindeki yeni kurallardan daha sık geçer. Kural sayısı episode sonunda ortalama 10,4; kendini ayarlayan tabanla (10,3) aynı, fark kural sayısından değil içeriğinden gelir.");


  // ============================================================ 7b Example of an admitted proposal
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Değerlendirme" });
  s.addText("Kabul edilen bir öneri örneği", { placeholder: "title" });
  card(s, M, 1.5, 7.2, 2.3, { fill: C.background2, name: "rule card" });
  txt(s, "Dil modelinin önerdiği kural (S3, dünya 201, dakika 2)", { x: M + 0.25, y: 1.6, w: 6.8, h: 0.4, fontSize: 15, color: C.accent4, italic: true });
  txt(s, [
    { text: "IF ", options: { bold: true, color: C.accent1 } }, { text: "asset exposure risk is VH ", options: {} },
    { text: "AND ", options: { bold: true, color: C.accent1 } }, { text: "suppression feasibility is VH", options: { breakLine: true } },
    { text: "THEN ", options: { bold: true, color: C.accent1 } }, { text: "preemptive asset collar 0.9", options: {} },
  ], { x: M + 0.25, y: 2.0, w: 6.8, h: 1.0, fontSize: 19, fontFace: "Consolas", color: C.text2 });
  txt(s, "Konumlama: su kaynağı (48,22), 30 m, hava aracı, ≈15 dk. Kaynak: üretken (LLM), kapılar G1-G4 geçildi; kural tabanına adaptation-born etiketiyle işlendi.", { x: M + 0.25, y: 3.05, w: 6.8, h: 0.7, fontSize: 14, color: C.text1 });
  // clauses
  card(s, M, 3.95, 7.2, 2.4, { name: "clauses card" });
  txt(s, "Yeni müdahale: doğrulanmış etkilerden kurulan 3 koşul", { x: M + 0.25, y: 4.05, w: 6.8, h: 0.4, fontSize: 15, color: C.accent4, italic: true });
  const clauses = [
    [{ text: "Etki", options: sh }, { text: "Sektör", options: sh }, { text: "Hücre bandı", options: sh }, { text: "Miktar", options: sh }],
    ["kaplama (coat)", "varlıklar", "0-3", "1,0"],
    ["kaplama (coat)", "yerleşim", "0-2", "1,0"],
    ["hazırlama (prime)", "yerleşim", "0-2", "0,8"],
  ].map((r, i) => i === 0 ? r : r.map((t) => ({ text: t, options: sc })));
  s.addTable(clauses, { x: M + 0.25, y: 4.5, w: 6.7, colW: [2.2, 1.6, 1.5, 1.4], rowH: 0.42, border: { type: "solid", color: H.accent5, pt: 0.75 }, fontFace: "+mn-lt", objectName: "clauses table" });
  // gate verdict
  card(s, 8.1, 1.5, 4.65, 4.85, { fill: C.background2, name: "gate card" });
  txt(s, "Kapı kararı", { x: 8.4, y: 1.65, w: 4.1, h: 0.45, fontSize: 21, bold: true, color: C.accent1 });
  const gv = [["G1, G2", "şema ve sözlük denetimi: geçti"], ["G3", "gölge tahmin Ĵ: 0,1351 → 0,1284 (öneriyle daha düşük)"], ["G4", "ikinci tohumla aynı yön: geçti"], ["G5", "paket marjı: ΔĴ = −0,0067, eşik aşıldı"], ["G5b", "tek nesne, katkı ayrımı gerekmedi"]];
  y = 2.2;
  for (const [g, t] of gv) {
    txt(s, g, { x: 8.4, y, w: 0.95, h: 0.7, fontSize: 17, bold: true, color: C.text2, valign: "top" });
    txt(s, t, { x: 9.35, y, w: 3.25, h: 0.7, fontSize: 15, valign: "top" });
    y += 0.76;
  }
  txt(s, "Aynı önerinin reddi de aynı biçimde kayıtlıdır: kapı adı, iki tahmin değeri, gerekçe.", { x: 8.4, y: 6.0, w: 4.2, h: 0.3, fontSize: 13, color: C.accent4, italic: true });
  s.addNotes("6:00-6:20. Somut bir örnek. Kent-orman arayüzü senaryosunda ikinci dakikada model, varlık maruziyeti ve söndürme uygulanabilirliği çok yüksekken önleyici bir varlık kuşağı öneriyor. Bu başlangıç kurallarında olmayan bir taktik: yangın gelmeden varlık ve yerleşim hücrelerini kaplamak ve hazırlamak. Öneri üç koşuldan kuruluyor; her koşul benzetimin bildiği bir etki, bir sektör ve bir hücre bandı. Gölge tahmin öneriyle maliyeti 0,135'ten 0,128'e indiriyor, ikinci tohum aynı yönü veriyor, paket marjı aşılıyor; kural kabul ediliyor ve kaynağı, kapıları, konumuyla birlikte deftere işleniyor. Red de aynı ayrıntıyla kayıtlı.");

  // ============================================================ 8a Burned area vs no DSS
  const SC = ["S1", "S2", "S3", "S4", "S5"];
  const chartBase = {
    barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
    dataLabelFontSize: 13, dataLabelColor: H.dk1, dataLabelFontFace: "+mn-lt",
    catAxisLabelFontSize: 15, catAxisLabelColor: H.dk1, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontSize: 12, valAxisLabelColor: H.accent4, valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: H.accent5, size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: H.dk1, showTitle: false,
  };
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Değerlendirme" });
  s.addText("Yanan alan: DisasterAware statik tabana göre %44 az", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "KDS yok", labels: SC, values: [269.3, 269.6, 233.5, 240.5, 240.5] },
    { name: "F5 statik", labels: SC, values: [89.2, 186.5, 27.4, 48.1, 53.4] },
    { name: "DisasterAware", labels: SC, values: [36.7, 120.6, 14.1, 27.2, 29.2] },
  ], Object.assign({ x: M, y: 1.5, w: 7.9, h: 5.0, chartColors: [H.accent4, H.accent5, H.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMaxVal: 300,
    valAxisTitle: "yanan alan (ha)", showValAxisTitle: true, valAxisTitleFontSize: 13, valAxisTitleColor: H.accent4, valAxisTitleFontFace: "+mn-lt",
    objectName: "burned area chart" }, chartBase));
  const tilesA = [["251 → 81 → 46 ha", "ortalama yanan alan: KDS yok, F5 statik, DisasterAware", H.accent1],
                  ["−%76", "etkilenen nüfus, statik tabana karşı DisasterAware (494 → 117 kişi)", H.dk2],
                  ["0,598 → 0,155 → 0,081", "fiziksel maliyet Jphys", H.dk2]];
  y = 1.5;
  for (const [n, l, col] of tilesA) {
    card(s, 8.85, y, 3.9, 1.55, { name: "tile " + l });
    txt(s, n, { x: 9.1, y: y + 0.1, w: 3.5, h: 0.75, fontSize: n.length > 14 ? 24 : 30, bold: true, color: col, valign: "middle" });
    txt(s, l, { x: 9.1, y: y + 0.85, w: 3.5, h: 0.6, fontSize: 15, color: C.text1 });
    y += 1.7;
  }
  s.addNotes("6:15-6:45. İki katman var. Kural tabanlı çekirdek (F5) tek başına yanan alanı serbest yanmaya göre yüzde 68 düşürür; bu kazanç kural tabanının kendisine aittir. Açık karar uzayı bunun üzerine gelir: DisasterAware statik tabana göre yanan alanı yüzde 44 daha düşürür (eşli fark 35 ± 9 ha, 150 dünyanın 131'inde daha iyi) ve beş senaryonun hepsinde en düşük sütundur; S1'de 89'dan 37 hektara, S3'te 27'den 14'e. En zor senaryo S2 (kısıtlı kaynak, iki uzak tutuşma): 187'den 121'e. Kendini ayarlayan tabana (F5+EvFIS, 75 ha) göre de 30 ± 7 ha, 150 dünyanın 128'inde; sorulursa söyle.");

  // ============================================================ 8b Adaptation contribution
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Değerlendirme" });
  s.addText("Söndürülen yangın oranı: %13'ten %75'e", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "F5 statik", labels: SC.concat(["Ort."]), values: [20, 3, 20, 10, 10, 13] },
    { name: "DisasterAware", labels: SC.concat(["Ort."]), values: [70, 23, 87, 97, 100, 75] },
  ], Object.assign({ x: M, y: 1.5, w: 7.9, h: 5.0, chartColors: [H.accent5, H.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMaxVal: 100,
    valAxisTitle: "olay süresinde söndürülen yangın (%)", showValAxisTitle: true, valAxisTitleFontSize: 13, valAxisTitleColor: H.accent4, valAxisTitleFontFace: "+mn-lt",
    objectName: "extinguished chart" }, chartBase));
  const tilesB = [["%13 → %75", "olay süresi içinde söndürülen yangın, statik tabana karşı DisasterAware", H.accent1],
                  ["131 / 150", "eşleştirilmiş dünyada daha az yanan alan; beş senaryonun hepsinde daha az nüfus, daha yüksek söndürme", H.dk2],
                  ["S4: %97, S5: %100", "kısıtlı kaynaklı arayüz senaryoları, gözlem kesintisi altında bile", H.dk2]];
  y = 1.5;
  for (const [n, l, col] of tilesB) {
    card(s, 8.85, y, 3.9, 1.55, { name: "tile " + l });
    txt(s, n, { x: 9.1, y: y + 0.1, w: 3.5, h: 0.75, fontSize: n.length > 12 ? 24 : 30, bold: true, color: col, valign: "middle" });
    txt(s, l, { x: 9.1, y: y + 0.85, w: 3.5, h: 0.65, fontSize: 15, color: C.text1 });
    y += 1.7;
  }
  s.addNotes("6:45-7:15. Açık karar uzayının katkısı en net söndürme oranında: statik tabanda yüzde 13, DisasterAware ile yüzde 75. Kent-orman arayüzü senaryolarında yüzde 87 ile 100; S5'te gözlem kesintisi ve gürültü altında bile yüzde 100, yanan alan S4'e göre yalnızca yüzde 7 artar. S2 en zor senaryo: kısıtlı kaynak ve iki uzak tutuşma; statik taban yüzde 3, DisasterAware yüzde 23. Kazanç yeni kurallardan ve sektör + hücre bandı ile konumlandırılmış müdahalelerden gelir (episode başına ortalama 39 konumlandırılmış emir); başlangıç kural kümesinde bunların hiçbiri yoktur ve her biri kapı dizisinden geçerek kabul edilmiştir. Bedel hesaplamadır: episode başına 10 model çağrısı, 124 s; kaynak maliyeti statik tabanın altında.");

  // ============================================================ 9 Generality and limits
  pres.addSection({ title: "Sonuç" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Sonuç" });
  s.addText("Genellik ve sınırlar", { placeholder: "title" });
  txt(s, "Dört koşul yeterli", { x: M, y: 1.6, w: 6, h: 0.5, fontSize: 22, bold: true, color: C.text2 });
  const conds = [["FaLayerGroup", "Sınırlı boyutlu durum soyutlaması"], ["FaBook", "Kapalı bir müdahale sözlüğü"], ["FaCubes", "Karar mantığından ayrık gölge benzetim veya sayısal ikiz"], ["FaCalculator", "Sayısal bir maliyet ölçüsü"]];
  y = 2.2;
  for (const [ic, t] of conds) {
    await circleIcon(s, ic, M, y, 0.7, H.dk2, H.lt1);
    txt(s, t, { x: M + 0.9, y, w: 5.2, h: 0.7, fontSize: 19, valign: "middle" });
    y += 0.92;
  }
  txt(s, "Sel tahliyesi, süreç güvenliği, şebeke yeniden yapılandırması, lojistik: aynı döngü, aynı kapılar; alana özgü olan yalnızca sözlük, maliyet terimleri ve benzetim.", { x: M, y: 5.95, w: 6.3, h: 0.9, fontSize: 15, color: C.accent4, italic: true });
  card(s, 7.5, 1.6, 5.25, 4.9, { fill: C.background2, name: "limits card" });
  txt(s, "İki sınır", { x: 7.8, y: 1.85, w: 4.6, h: 0.5, fontSize: 22, bold: true, color: C.accent1 });
  txt(s, [
    { text: "Fiziksel etki kümesi öneriye kapalı", options: { bold: true, color: C.text2, breakLine: true } },
    { text: "Yeni bir etki, benzetimde doğrulanmış yeni bir alan gerektirir ve tasarımcı ekler; aksi halde benzetimin güvence işlevi kalmaz.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Tek tek öneriler birebir yeniden üretilemez", options: { bold: true, color: C.text2, breakLine: true } },
    { text: "Yeniden üretilebilir olan süreçtir: model, istem şablonu ve tohum tasarımı sabit; kabul kararlarının tamamı belirlenimci ve kayıtlı.", options: {} },
  ], { x: 7.8, y: 2.45, w: 4.65, h: 3.9, fontSize: 17 });
  s.addNotes("7:15-7:35. Mimaride yangına özgü bileşen yok. Dört koşulu sağlayan her KDS'de aynı döngü ve kapı dizisi değişmeden kullanılabilir. İki sınır: fiziksel etki kümesi tasarımcıya ait; önerici deterministik değil, süreç deterministik.");

  // ============================================================ 10 Conclusion
  s = pres.addSlide({ masterName: "CONTENT_DARK", sectionTitle: "Sonuç" });
  s.addText("Sonuç", { placeholder: "title" });
  const concl = [
    ["FaDoorOpen", "Karar uzayı açılır", "Parametre ayarı ve çözünürlük büyütme mevcut bilgiyi esnetir; şema kısıtlı LLM yeni kural, kavram ve bileşik müdahale ekler."],
    ["FaShieldAlt", "Güven doğrulamadan gelir", "Belirlenimci kapı dizisi, çift tohumlu benzetim denetimi ve yeterlilik temelli kabul; model eğitimi değil."],
    ["FaChartLine", "Seçici ve etkili", "%22-43 kabul; söndürülen yangın oranı %13'ten %75'e, yanan alan statik tabana göre %44 az, 150 eşleştirilmiş dünyanın 131'inde."],
  ];
  x = M;
  for (const [ic, h1, h2] of concl) {
    await circleIcon(s, ic, x, 1.7, 0.95, H.accent1, H.lt1);
    txt(s, h1, { x, y: 2.85, w: 3.75, h: 0.6, fontSize: 24, bold: true, color: C.background1 });
    txt(s, h2, { x, y: 3.5, w: 3.75, h: 2.2, fontSize: 17, color: C.accent5 });
    x += 4.1;
  }
  txt(s, "Gelecek çalışma: farklı afet türleri ve endüstriyel KDS'lere aktarım; operatör güveninin saha çalışmalarıyla ölçülmesi.", { x: M, y: 5.95, w: 12.1, h: 0.6, fontSize: 17, color: C.accent3, italic: true });
  txt(s, "Teşekkürler. caglar.akman@metu.edu.tr", { x: M, y: 6.5, w: 12.1, h: 0.45, fontSize: 17, color: C.background1 });
  s.addNotes("7:35-8:00. Üç cümlelik kapanış ve gelecek çalışma. Sorulara geç.");

  await pres.writeFile({ fileName: "TOK2026_AcikKararUzayi_Sunum_v4.pptx" });
  await applyTheme("TOK2026_AcikKararUzayi_Sunum_v4.pptx", THEME);
  console.log("done");
})();
