"""Manuscript Figure 1: Energy Hub physical superstructure (catalog -> buses -> demands).

Emits, for two label modes, an editable draw.io file and a matching SVG (converted to
PDF/PNG) from one shared shape/edge list:
  - "equations" -- balance/constraint formulas on the arrows and legend;
  - "text_only" -- same nodes and connections, plain descriptive labels only.

Catalog values (family, eta_E, eta_Q, eta_C) are read from
code/hybrid_pipeline/benchmarks/energy_hub/instance.py.
"""

import os
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent / "main" / "figure1_energy_hub_superstructure")

CATALOG = [
    ("tech_01", "cogen", 0.46, 0.42, 0.12),
    ("tech_02", "cogen", 0.42, 0.48, 0.08),
    ("tech_03", "cogen", 0.38, 0.55, 0.06),
    ("tech_04", "cogen", 0.50, 0.30, 0.10),
    ("tech_05", "cogen", 0.34, 0.62, 0.05),
    ("tech_06", "cogen", 0.40, 0.36, 0.15),
    ("tech_07", "renewable", 0.95, 0.05, 0.02),
    ("tech_08", "renewable", 0.90, 0.08, 0.03),
    ("tech_09", "renewable", 0.88, 0.12, 0.04),
    ("tech_10", "renewable", 0.82, 0.04, 0.02),
    ("tech_11", "thermal", 0.05, 0.92, 0.10),
    ("tech_12", "thermal", 0.04, 0.88, 0.06),
    ("tech_13", "thermal", 0.03, 0.84, 0.08),
    ("tech_14", "thermal", 0.02, 0.78, 0.04),
    ("tech_15", "cooling", 0.10, 0.08, 0.95),
    ("tech_16", "cooling", 0.08, 0.10, 0.88),
    ("tech_17", "cooling", 0.06, 0.06, 0.82),
    ("tech_18", "cooling", 0.04, 0.04, 0.76),
]

DOM_WORD = {"E": "electricity", "Q": "heat", "C": "cooling"}


def per_tech_dominant(e, q, c):
    return max((("E", e), ("Q", q), ("C", c)), key=lambda x: x[1])[0]


FAMILIES = {
    "cogen": dict(label="Cogeneration family (6 candidates)", techs=[c for c in CATALOG if c[1] == "cogen"], y=120, h=175, dominant=["E", "Q"]),
    "renewable": dict(label="Renewable family (4 candidates)", techs=[c for c in CATALOG if c[1] == "renewable"], y=315, h=140, dominant=["E"]),
    "thermal": dict(label="Thermal family (4 candidates)", techs=[c for c in CATALOG if c[1] == "thermal"], y=475, h=140, dominant=["Q"]),
    "cooling": dict(label="Cooling family (4 candidates)", techs=[c for c in CATALOG if c[1] == "cooling"], y=635, h=140, dominant=["C"]),
}

BUS_COLOR = {
    "E": dict(fill="#DAE8FC", stroke="#6C8EBF", name="Electricity bus  E"),
    "Q": dict(fill="#FFE6CC", stroke="#D79B00", name="Heat bus  Q"),
    "C": dict(fill="#D5E8D4", stroke="#82B366", name="Cooling bus  C"),
}
BUS_Y = {"E": (120, 230), "Q": (390, 190), "C": (620, 150)}

CANVAS_W, CANVAS_H = 2020, 1000

NEUTRAL_FILL, NEUTRAL_STROKE = "#F5F5F5", "#666666"
INPUT_FILL, INPUT_STROKE = "#FFF2CC", "#D6B656"
DEMAND_FILL, DEMAND_STROKE = "#F8CECC", "#B85450"
BANNER_FILL, BANNER_STROKE = "#E1D5E7", "#9673A6"
DOM_COLOR = {"E": "#6C8EBF", "Q": "#D79B00", "C": "#82B366"}

TEXT = {
    "equations": dict(
        banner="Exact-cardinality support:   Sum_i y_i = 6   (activates 6 of 18 candidate technologies)",
        legend=("For each candidate technology i:\n"
                "y_i in {0,1}  --  selected in support        s_i >= 0  --  installed capacity        P_i,t >= 0  --  period dispatch\n"
                "s_i <= S_i^max y_i          P_i,t <= rho_i,t s_i"),
        grid_label="G_t <= 0.4 D_t^E",
        dem_label={"E": "Sum_i eta_i^E P_i,t + G_t = D_t^E", "Q": "Sum_i eta_i^Q P_i,t >= D_t^Q", "C": "Sum_i eta_i^C P_i,t >= D_t^C"},
        chip=lambda tid, e, q, c: f"{tid.replace('tech_', 't')}\nE={e:.2f}  Q={q:.2f}  C={c:.2f}",
    ),
    "text_only": dict(
        banner="Exact-cardinality support: exactly 6 of the 18 candidate technologies are active at a time",
        legend=("For each candidate technology:\n"
                "a yes/no selection decision, an installed-capacity decision bounded by the technology's\n"
                "maximum size, and a period-by-period operating-level decision bounded by installed capacity and availability"),
        grid_label="Grid can cover up to 40% of electricity demand",
        dem_label={"E": "Electricity demand must be matched exactly", "Q": "Heat demand must be covered (minimum)", "C": "Cooling demand must be covered (minimum)"},
        chip=lambda tid, e, q, c: f"{tid.replace('tech_', 't')}\ndominant: {DOM_WORD[per_tech_dominant(e, q, c)]}",
    ),
}


def build(mode):
    txt = TEXT[mode]
    shapes = []
    edges = []
    _id = [0]

    def nid():
        _id[0] += 1
        return f"n{_id[0]}"

    def add_shape(x, y, w, h, label, fill, stroke, fontsize=13, bold=False, align="middle", chip=False):
        sid = nid()
        shapes.append(dict(id=sid, x=x, y=y, w=w, h=h, label=label, fill=fill, stroke=stroke,
                            fontsize=fontsize, bold=bold, align=align, chip=chip))
        return sid

    def add_edge(src, tgt, label="", color="#666666", width=2, dashed=False, fontsize=12):
        edges.append(dict(src=src, tgt=tgt, label=label, color=color, width=width, dashed=dashed, fontsize=fontsize))

    add_shape(380, 15, 760, 44, txt["banner"], BANNER_FILL, BANNER_STROKE, fontsize=14, bold=True)

    in_primary = add_shape(40, 150, 260, 70, "Dispatchable primary\nenergy input", INPUT_FILL, INPUT_STROKE)
    in_renew = add_shape(40, 320, 260, 70, "Renewable resource availability\n(time-varying)", INPUT_FILL, INPUT_STROKE)
    in_grid = add_shape(40, 490, 260, 70, "Grid electricity", INPUT_FILL, INPUT_STROKE)

    CATALOG_FRAME = dict(x=380, y=75, w=760, h=770,
                          label="Candidate technology catalog -- N = 18  (6 cogeneration + 4 renewable + 4 thermal + 4 cooling)")

    fam_ids = {}
    for key, fam in FAMILIES.items():
        fam_id = add_shape(410, fam["y"], 700, fam["h"], fam["label"], NEUTRAL_FILL, NEUTRAL_STROKE, fontsize=13, bold=True, align="top")
        fam_ids[key] = fam_id
        n = len(fam["techs"])
        chip_w = 640 / n - 10
        for i, (tid, _, e, q, c) in enumerate(fam["techs"]):
            cx = 430 + i * (chip_w + 12)
            cy = fam["y"] + 55
            add_shape(cx, cy, chip_w, 80, txt["chip"](tid, e, q, c), "#FFFFFF", "#666666", fontsize=10, chip=True)

    add_shape(380, 862, 760, 100, txt["legend"], "#FFFFFF", "#999999", fontsize=12, align="top")

    bus_ids = {}
    for b, (y, h) in BUS_Y.items():
        col = BUS_COLOR[b]
        bus_ids[b] = add_shape(1220, y, 140, h, col["name"], col["fill"], col["stroke"], fontsize=14, bold=True)

    dem_ids = {}
    dem_specs = {"E": (190, "Electricity demand\nD_t^E" if mode == "equations" else "Electricity demand"),
                 "Q": (440, "Heat demand\nD_t^Q" if mode == "equations" else "Heat demand"),
                 "C": (650, "Cooling demand\nD_t^C" if mode == "equations" else "Cooling demand")}
    for b, (y, label) in dem_specs.items():
        dem_ids[b] = add_shape(1660, y, 300, 90, label, DEMAND_FILL, DEMAND_STROKE, fontsize=13, bold=True)

    add_edge(in_primary, fam_ids["cogen"], color="#666666", width=2)
    add_edge(in_primary, fam_ids["thermal"], color="#666666", width=2)
    add_edge(in_primary, fam_ids["cooling"], color="#666666", width=2)
    add_edge(in_renew, fam_ids["renewable"], color="#666666", width=2)

    add_edge(in_grid, bus_ids["E"], label=txt["grid_label"], color="#6C8EBF", width=3)

    for key, fam in FAMILIES.items():
        for b in ("E", "Q", "C"):
            if b in fam["dominant"]:
                add_edge(fam_ids[key], bus_ids[b], color=DOM_COLOR[b], width=3)
            else:
                add_edge(fam_ids[key], bus_ids[b], color="#AAAAAA", width=1, dashed=True)

    add_edge(bus_ids["E"], dem_ids["E"], label=txt["dem_label"]["E"], color="#6C8EBF", width=3)
    add_edge(bus_ids["Q"], dem_ids["Q"], label=txt["dem_label"]["Q"], color="#D79B00", width=3)
    add_edge(bus_ids["C"], dem_ids["C"], label=txt["dem_label"]["C"], color="#82B366", width=3)

    return shapes, edges, CATALOG_FRAME, nid


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def esc_html(s):
    return "&lt;br&gt;".join(esc(line) for line in str(s).split("\n"))


def drawio_style(s):
    align = "middle" if s["align"] == "middle" else "top"
    style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={s['fill']};strokeColor={s['stroke']};"
             f"fontSize={s['fontsize']};align=center;verticalAlign={align};")
    if s["bold"]:
        style += "fontStyle=1;"
    return style


def render_drawio(shapes, edges, catalog_frame, nid, out_path):
    cells_xml = []
    frame_id = nid()
    cells_xml.append(
        f'<mxCell id="{frame_id}" value="{esc_html(catalog_frame["label"])}" '
        'style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#000000;fontSize=14;fontStyle=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;dashed=0;" '
        f'vertex="1" parent="1"><mxGeometry x="{catalog_frame["x"]}" y="{catalog_frame["y"]}" '
        f'width="{catalog_frame["w"]}" height="{catalog_frame["h"]}" as="geometry"/></mxCell>'
    )
    for s in shapes:
        cells_xml.append(
            f'<mxCell id="{s["id"]}" value="{esc_html(s["label"])}" style="{drawio_style(s)}" vertex="1" parent="1">'
            f'<mxGeometry x="{s["x"]}" y="{s["y"]}" width="{s["w"]}" height="{s["h"]}" as="geometry"/></mxCell>'
        )
    for e in edges:
        style = f"endArrow=block;html=1;strokeColor={e['color']};strokeWidth={e['width']};"
        if e["dashed"]:
            style += "dashed=1;"
        if e["label"]:
            style += f"fontColor={e['color']};fontSize={e['fontsize']};"
        cells_xml.append(
            f'<mxCell id="{nid()}" value="{esc_html(e["label"])}" style="{style}" edge="1" parent="1" source="{e["src"]}" target="{e["tgt"]}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>'
        )

    xml = (
        '<mxfile host="app.diagrams.net">'
        '<diagram name="Energy Hub Superstructure">'
        f'<mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{CANVAS_W}" pageHeight="{CANVAS_H}" math="0" shadow="0">'
        '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
        + "".join(cells_xml) +
        '</root></mxGraphModel></diagram></mxfile>'
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print("Wrote", out_path, len(xml), "bytes")

    import xml.etree.ElementTree as ET
    ET.parse(out_path)
    print("  -> XML well-formed OK")


def svg_esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap_text_svg(label, x, y, w, h, fontsize, bold, align):
    lines = label.split("\n")
    weight = "bold" if bold else "normal"
    n = len(lines)
    if align == "top":
        start_y = y + fontsize + 6
        line_h = fontsize + 5
        anchor = "start"
        tx = x + 8
    else:
        line_h = fontsize + 4
        start_y = y + h / 2 - (n - 1) * line_h / 2 + fontsize / 3
        anchor = "middle"
        tx = x + w / 2
    out = []
    for i, ln in enumerate(lines):
        out.append(
            f'<text x="{tx:.1f}" y="{start_y + i * line_h:.1f}" font-size="{fontsize}" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="#111111">{svg_esc(ln)}</text>'
        )
    return "".join(out)


def render_svg(shapes, edges, catalog_frame, out_path):
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" font-family="Helvetica,Arial,sans-serif">',
        f'<rect x="0" y="0" width="{CANVAS_W}" height="{CANVAS_H}" fill="#FFFFFF"/>',
        '<defs>'
    ]
    for name, color in [("arrowhead", "#000000")] + [(f"arrow_{c.strip('#')}", c) for c in
            {"#6C8EBF", "#D79B00", "#82B366", "#666666", "#AAAAAA", "#9673A6"}]:
        svg_parts.append(
            f'<marker id="{name}" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M0,0 L0,6 L9,3 z" fill="{color}"/></marker>'
        )
    svg_parts.append('</defs>')

    svg_parts.append(
        f'<rect x="{catalog_frame["x"]}" y="{catalog_frame["y"]}" width="{catalog_frame["w"]}" height="{catalog_frame["h"]}" '
        f'fill="none" stroke="#000000" stroke-width="1.5"/>'
    )
    svg_parts.append(
        f'<text x="{catalog_frame["x"]+10}" y="{catalog_frame["y"]+20}" font-size="14" font-weight="bold" fill="#111111">'
        f'{svg_esc(catalog_frame["label"])}</text>'
    )

    shape_by_id = {s["id"]: s for s in shapes}
    label_jobs = []
    for e in edges:
        s1 = shape_by_id[e["src"]]
        s2 = shape_by_id[e["tgt"]]
        x1, y1 = s1["x"] + s1["w"], s1["y"] + s1["h"] / 2
        x2, y2 = s2["x"], s2["y"] + s2["h"] / 2
        dash = ' stroke-dasharray="6,5"' if e["dashed"] else ""
        mid_x = (x1 + x2) / 2
        path = f'M {x1:.1f},{y1:.1f} C {mid_x:.1f},{y1:.1f} {mid_x:.1f},{y2:.1f} {x2:.1f},{y2:.1f}'
        marker = {"#6C8EBF": "arrow_6C8EBF", "#D79B00": "arrow_D79B00", "#82B366": "arrow_82B366",
                  "#666666": "arrow_666666", "#AAAAAA": "arrow_AAAAAA"}.get(e["color"], "arrowhead")
        svg_parts.append(
            f'<path d="{path}" fill="none" stroke="{e["color"]}" stroke-width="{e["width"]}"{dash} '
            f'marker-end="url(#{marker})"/>'
        )
        if e["label"]:
            lx = mid_x
            fx0, fx1 = catalog_frame["x"], catalog_frame["x"] + catalog_frame["w"]
            if fx0 <= lx <= fx1:
                lx = fx1 + 40
            label_jobs.append(dict(x=lx, y=(y1 + y2) / 2 - 6, label=e["label"], color=e["color"], fontsize=e["fontsize"]))

    for s in shapes:
        rx = 6 if not s["chip"] else 4
        stroke_w = 1.2 if s["chip"] else 1.6
        svg_parts.append(
            f'<rect x="{s["x"]}" y="{s["y"]}" width="{s["w"]:.1f}" height="{s["h"]}" rx="{rx}" '
            f'fill="{s["fill"]}" stroke="{s["stroke"]}" stroke-width="{stroke_w}"/>'
        )
        svg_parts.append(wrap_text_svg(s["label"], s["x"], s["y"], s["w"], s["h"], s["fontsize"], s["bold"], s["align"]))

    for j in label_jobs:
        w = max(120, 7.2 * len(j["label"]))
        svg_parts.append(
            f'<rect x="{j["x"]-w/2:.1f}" y="{j["y"]-13:.1f}" width="{w:.1f}" height="17" fill="#FFFFFF" opacity="0.92" stroke="#DDDDDD" stroke-width="0.5"/>'
            f'<text x="{j["x"]:.1f}" y="{j["y"]:.1f}" font-size="{j["fontsize"]}" text-anchor="middle" fill="{j["color"]}">{svg_esc(j["label"])}</text>'
        )

    svg_parts.append('</svg>')
    svg = "\n".join(svg_parts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(svg)
    print("Wrote", out_path, len(svg), "bytes")


OUT_DIRS = {
    "equations": BASE_DIR,
    "text_only": os.path.join(BASE_DIR, "variant_text_only"),
}
STEMS = {
    "equations": "Figure0_energy_hub_superstructure",
    "text_only": "Figure0_energy_hub_superstructure_text_only",
}

for mode in ("equations", "text_only"):
    out_dir = OUT_DIRS[mode]
    os.makedirs(out_dir, exist_ok=True)
    shapes, edges, catalog_frame, nid = build(mode)
    print(f"[{mode}] {len(shapes)} shapes, {len(edges)} edges")
    render_drawio(shapes, edges, catalog_frame, nid, os.path.join(out_dir, STEMS[mode] + ".drawio"))
    render_svg(shapes, edges, catalog_frame, os.path.join(out_dir, STEMS[mode] + ".svg"))
