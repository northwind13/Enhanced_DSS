/**
 * Export everything the DSS has GENERATED or MODIFIED into a Word table.
 *
 * The learned store (logs/dss_generated_state.json) is the system's own
 * record of what its adaptation stages produced: rules the resolution stage
 * instantiated, rules the generative stage wrote, consequents evFIS tuned,
 * membership boundaries it shifted, and the concepts and interventions the
 * generative stage added to the vocabulary. It is append-only and carries
 * the provenance of every record, which is what makes a table like this
 * possible at all.
 *
 * THE DOCUMENT IS BUILT FROM THE STORE, never typed. A hand-kept list of
 * learned rules is a second store that drifts from the first.
 *
 *   node validation/export_learned_rules.js
 *   node validation/export_learned_rules.js --store ../other/state.json \
 *        --out learned_rules.docx
 *
 * Requires the `docx` npm package (npm install docx).
 */

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, WidthType, AlignmentType, ShadingType, BorderStyle,
} = require("docx");

// ---------------------------------------------------------------- arguments
const argv = process.argv.slice(2);
const arg = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt;
};
const HERE = __dirname;
const STORE = path.resolve(arg("--store",
  path.join(HERE, "..", "logs", "dss_generated_state.json")));
const OUT = path.resolve(arg("--out",
  path.join(HERE, "figures", "learned_rules.docx")));

// ------------------------------------------------------------------ reading
const store = JSON.parse(fs.readFileSync(STORE, "utf8"));

/** "IF a is X AND b is Y THEN c 0.90, d 0.70" */
function ruleText(r) {
  const ants = (r.antecedents || [])
    .map(([v, t]) => `${String(v).replace(/_/g, " ")} is ${t}`).join(" AND ");
  const cons = (r.consequents || [])
    .map(([c, v]) => `${String(c).replace(/_/g, " ")} ${Number(v).toFixed(2)}`)
    .join(", ");
  return `IF ${ants} THEN ${cons}`;
}

function consText(list) {
  return (list || [])
    .map(([c, v]) => `${String(c).replace(/_/g, " ")} ${Number(v).toFixed(2)}`)
    .join(", ");
}

function when(rec) {
  const t = rec.trigger || {};
  const bits = [];
  if (t.step !== undefined) bits.push(`step ${t.step}`);
  if (t.minute !== undefined) bits.push(`${t.minute} min`);
  if (t.deficit !== undefined) bits.push(`deficit ${t.deficit}`);
  const ts = (rec.timestamp || "").replace("T", " ").replace("Z", "");
  return bits.join(", ") + (ts ? ` · ${ts}` : "");
}

// Every record becomes one row: {tip, id, kural, aciklama, asama, kaynak}
const rows = [];

for (const r of store.genai_rules || []) {
  rows.push({
    tip: "Generated rule (GenAI)",
    id: r.name || r.id,
    kural: ruleText(r),
    aciklama: `${r.note || ""} — ${when(r)}`,
    asama: `Stage ${r.source_stage}`,
    kaynak: r.id,
    seq: r.seq || 0,
  });
}

for (const m of store.evfis_rule_modifications || []) {
  const t = m.modification_type;
  if (t === "rule_add") {
    const r = (m.after && m.after.rule) || {};
    rows.push({
      tip: "Generated rule (resolution)",
      id: r.name || m.base_rule_id,
      kural: ruleText(r),
      aciklama: `${r.note || ""} — ${when(m)}`,
      asama: `Stage ${m.source_stage}`,
      kaynak: m.id,
      seq: m.seq || 0,
    });
  } else if (t === "consequent_update") {
    rows.push({
      tip: "Consequent tuning (evFIS)",
      id: m.base_rule_id,
      kural: `${consText(m.before && m.before.consequents)}  →  `
           + `${consText(m.after && m.after.consequents)}`,
      aciklama: `consequent values tuned — ${when(m)}`,
      asama: `Stage ${m.source_stage}`,
      kaynak: m.id,
      seq: m.seq || 0,
    });
  } else if (t === "membership_shift") {
    const b = m.before || {}, a = m.after || {};
    rows.push({
      tip: "Membership shift (evFIS)",
      id: `${m.base_rule_id} · ${a.variable || b.variable || ""}`,
      kural: JSON.stringify(a).slice(0, 220),
      aciklama: `shared boundary moved (Ruspini partition preserved) — ${when(m)}`,
      asama: `Stage ${m.source_stage}`,
      kaynak: m.id,
      seq: m.seq || 0,
    });
  } else if (t === "term_insert") {
    const a = m.after || {};
    rows.push({
      tip: "Term insertion (resolution)",
      id: `${a.variable || m.base_rule_id} · ${a.term || ""}`,
      kural: JSON.stringify(a).slice(0, 220),
      aciklama: `new linguistic term inserted into the variable — ${when(m)}`,
      asama: `Stage ${m.source_stage}`,
      kaynak: m.id,
      seq: m.seq || 0,
    });
  }
}

for (const c of store.genai_concepts || []) {
  const ins = (c.inputs || [])
    .map((i) => `${String(i.name).replace(/_/g, " ")} ×${i.weight}`).join(" + ");
  rows.push({
    tip: "Generated concept (GenAI)",
    id: c.name,
    kural: `${ins}  →  ${c.aggregation}`,
    aciklama: `layer ${c.layer} concept — ${when(c)}`,
    asama: `Stage ${c.source_stage}`,
    kaynak: c.id,
    seq: c.seq || 0,
  });
}

for (const iv of store.genai_interventions || []) {
  const comp = (iv.composition || [])
    .map((k) => `${String(k.channel).replace(/_/g, " ")} ×${k.weight}`)
    .join(" + ");
  rows.push({
    tip: "Generated intervention (GenAI)",
    id: iv.name,
    kural: comp,
    aciklama: `weighted composition of the base channels — ${when(iv)}`,
    asama: `Stage ${iv.source_stage}`,
    kaynak: iv.id,
    seq: iv.seq || 0,
  });
}

// grouped by kind, and within a kind in the order they were produced: the
// store's seq is the system's own clock, so the table reads as a history
const TYPE_ORDER = [
  "Generated rule (GenAI)",
  "Generated rule (resolution)",
  "Generated concept (GenAI)",
  "Generated intervention (GenAI)",
  "Consequent tuning (evFIS)",
  "Membership shift (evFIS)",
  "Term insertion (resolution)",
];
rows.sort((a, b) => {
  const d = TYPE_ORDER.indexOf(a.tip) - TYPE_ORDER.indexOf(b.tip);
  return d !== 0 ? d : a.seq - b.seq;
});

// -------------------------------------------------------------------- docx
const W = [1700, 1300, 4600, 3400, 900, 1700];   // DXA, sums to 13600
const TOTAL = W.reduce((a, b) => a + b, 0);

const cell = (text, opts = {}) => new TableCell({
  width: { size: opts.w, type: WidthType.DXA },
  shading: opts.head
    ? { type: ShadingType.CLEAR, fill: "D9E2F3" }
    : undefined,
  children: [new Paragraph({
    children: [new TextRun({
      text: String(text == null ? "" : text),
      bold: !!opts.head,
      font: opts.mono ? "Consolas" : undefined,
      size: opts.head ? 18 : 16,          // half-points: 9 pt / 8 pt
    })],
  })],
});

const header = new TableRow({
  tableHeader: true,
  children: ["Type", "ID", "Rule", "Description", "Stage", "Record"]
    .map((t, i) => cell(t, { w: W[i], head: true })),
});

const bodyRows = rows.map((r) => new TableRow({
  children: [
    cell(r.tip, { w: W[0] }),
    cell(r.id, { w: W[1], mono: true }),
    cell(r.kural, { w: W[2], mono: true }),
    cell(r.aciklama, { w: W[3] }),
    cell(r.asama, { w: W[4] }),
    cell(r.kaynak, { w: W[5], mono: true }),
  ],
}));

// a count per kind, so the reader knows the shape of what follows
const counts = {};
for (const r of rows) counts[r.tip] = (counts[r.tip] || 0) + 1;
const summary = new Table({
  columnWidths: [6000, 1600],
  width: { size: 7600, type: WidthType.DXA },
  rows: [
    new TableRow({
      tableHeader: true,
      children: [cell("Type", { w: 6000, head: true }),
                 cell("Count", { w: 1600, head: true })],
    }),
    ...TYPE_ORDER.filter((t) => counts[t]).map((t) => new TableRow({
      children: [cell(t, { w: 6000 }), cell(counts[t], { w: 1600 })],
    })),
    new TableRow({
      children: [cell("TOTAL", { w: 6000, head: true }),
                 cell(rows.length, { w: 1600, head: true })],
    }),
  ],
});

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 16838, height: 11906 },     // A4 landscape (DXA)
        margin: { top: 720, bottom: 720, left: 720, right: 720 },
      },
    },
    children: [
      new Paragraph({
        text: "DisasterAware — generated and modified rules",
        heading: HeadingLevel.HEADING_1,
      }),
      new Paragraph({
        children: [new TextRun({
          text: `Source: ${path.basename(STORE)} · `
              + `${rows.length} records · `
              + `${new Date().toISOString().slice(0, 10)}`,
          italics: true, size: 18,
        })],
      }),
      new Paragraph({
        children: [new TextRun({
          text: "This table is produced from the DSS's own learning "
              + "record: the rules the resolution stage instantiated, the "
              + "rules the generative stage wrote, the consequents evFIS "
              + "tuned and the membership boundaries it shifted, and the "
              + "concepts and interventions added to the vocabulary. "
              + "Records are grouped by kind and, within a kind, listed in "
              + "the order the system produced them.",
          size: 18,
        })],
        spacing: { after: 200 },
      }),
      new Paragraph({ text: "Summary", heading: HeadingLevel.HEADING_2 }),
      summary,
      new Paragraph({ text: "", spacing: { after: 200 } }),
      new Paragraph({ text: "Records", heading: HeadingLevel.HEADING_2 }),
      new Table({
        columnWidths: W,
        width: { size: TOTAL, type: WidthType.DXA },
        rows: [header, ...bodyRows],
      }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log(`${rows.length} records -> ${OUT}`);
  for (const t of TYPE_ORDER) if (counts[t]) console.log(`   ${counts[t]}  ${t}`);
});
