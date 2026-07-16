"""
OptiCrop — Crop Recommendation System
Flask app: ML prediction + crop info + soil log
"""

import os
import sqlite3
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "opticrop-dev-secret")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DB_PATH    = os.path.join(BASE_DIR, "opticrop.db")

# ── Load ML model ────────────────────────────────────────────────
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ── Crop metadata ────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"label":"Rice",         "emoji":"🌾","season":"Kharif (Jun–Nov)","duration":"110–140 days","soil":"Clay / Loam","fertilizer":"Urea, SSP, MOP","yield":"3.5–7 t/ha","pests":"Blast, Brown planthopper","use":"Food grain, starch, rice bran oil","desc":"Staple cereal requiring flooded paddy conditions. Thrives in humid tropical climates."},
    "maize":       {"label":"Maize",        "emoji":"🌽","season":"Kharif/Rabi","duration":"90–120 days","soil":"Well-drained Loam","fertilizer":"DAP, Urea, Potash","yield":"4–9 t/ha","pests":"Fall Armyworm, Stem borer","use":"Food, animal feed, ethanol","desc":"Versatile cereal grown globally for food, fodder, and industrial uses."},
    "chickpea":    {"label":"Chickpea",     "emoji":"🫘","season":"Rabi (Oct–Mar)","duration":"90–120 days","soil":"Sandy Loam","fertilizer":"DAP, Rhizobium","yield":"1–2.5 t/ha","pests":"Helicoverpa, Fusarium wilt","use":"Dal, flour, snacks","desc":"Drought-tolerant legume that fixes atmospheric nitrogen."},
    "kidneybeans": {"label":"Kidney Beans", "emoji":"🫘","season":"Kharif (Jun–Sep)","duration":"90–110 days","soil":"Loamy / Sandy Loam","fertilizer":"DAP, MOP, Rhizobium","yield":"1.2–2.5 t/ha","pests":"Bean Fly, Angular Leaf Spot","use":"Food (rajma), protein supplement","desc":"Popular legume rich in protein and fiber."},
    "pigeonpeas":  {"label":"Pigeon Peas",  "emoji":"🫘","season":"Kharif (Jun–Jan)","duration":"150–200 days","soil":"Medium to Heavy Soils","fertilizer":"SSP, MOP, Rhizobium","yield":"0.8–2.2 t/ha","pests":"Pod borer, Fusarium wilt","use":"Dal (toor), animal fodder","desc":"Perennial legume suited to semi-arid tropics."},
    "mothbeans":   {"label":"Moth Beans",   "emoji":"🫘","season":"Kharif (Jul–Sep)","duration":"60–85 days","soil":"Sandy (arid soils)","fertilizer":"SSP, MOP","yield":"0.5–1.5 t/ha","pests":"Pod borer, Aphids","use":"Dal, sprouts, fodder","desc":"Extremely drought-hardy legume native to arid regions."},
    "mungbean":    {"label":"Mung Bean",    "emoji":"🫘","season":"Kharif/Zaid","duration":"60–75 days","soil":"Sandy Loam","fertilizer":"DAP, Rhizobium","yield":"0.8–1.8 t/ha","pests":"Yellow Mosaic Virus, Aphids","use":"Dal (moong), sprouts","desc":"Short-duration legume ideal for intercropping."},
    "blackgram":   {"label":"Black Gram",   "emoji":"🫘","season":"Kharif/Rabi","duration":"70–90 days","soil":"Loamy / Clay Loam","fertilizer":"DAP, MOP, Rhizobium","yield":"0.7–1.5 t/ha","pests":"YMV, Pod borer","use":"Dal (urad), idli/dosa batter","desc":"Warm-season legume widely grown in South Asia."},
    "lentil":      {"label":"Lentil",       "emoji":"🫘","season":"Rabi (Oct–Mar)","duration":"100–130 days","soil":"Loam / Clay Loam","fertilizer":"DAP, Rhizobium","yield":"0.8–2 t/ha","pests":"Rust, Aphids","use":"Dal (masoor), soup","desc":"Cool-season legume thriving in temperate climates."},
    "pomegranate": {"label":"Pomegranate",  "emoji":"🍎","season":"Perennial (Jun / Feb)","duration":"5–7 months","soil":"Deep Loam","fertilizer":"FYM, NPK 15:15:15","yield":"15–30 t/ha","pests":"Anar butterfly, Fruit borer","use":"Fresh fruit, juice, export","desc":"Hardy subtropical fruit tree tolerating heat and mild drought."},
    "banana":      {"label":"Banana",       "emoji":"🍌","season":"Year-round (perennial)","duration":"11–14 months","soil":"Deep Rich Loam","fertilizer":"Urea, MOP, SSP, FYM","yield":"30–70 t/ha","pests":"Sigatoka, Panama wilt","use":"Fresh fruit, chips, export","desc":"Tropical monocot requiring abundant water and nutrients."},
    "mango":       {"label":"Mango",        "emoji":"🥭","season":"Perennial (Dec–May)","duration":"3–5 months","soil":"Deep Alluvial","fertilizer":"FYM, NPK, Ca, B","yield":"10–25 t/ha","pests":"Mango hopper, Fruit fly","use":"Fresh, pulp, pickle, export","desc":"King of fruits — thrives in hot dry summers and cool dry winters."},
    "grapes":      {"label":"Grapes",       "emoji":"🍇","season":"Perennial (Feb/Oct)","duration":"100–120 days","soil":"Sandy Loam / Gravel","fertilizer":"NPK fertigation, Ca, Mg, B","yield":"15–35 t/ha","pests":"Downy Mildew, Thrips","use":"Table grapes, raisins, wine","desc":"Temperate fruit crop requiring distinct seasons."},
    "watermelon":  {"label":"Watermelon",   "emoji":"🍉","season":"Zaid/Summer (Feb–Jun)","duration":"70–90 days","soil":"Sandy Loam","fertilizer":"DAP, Urea, MOP","yield":"20–50 t/ha","pests":"Aphids, Fusarium Wilt","use":"Fresh fruit, juice, seeds","desc":"Warm-season cucurbit thriving in hot summers."},
    "muskmelon":   {"label":"Muskmelon",    "emoji":"🍈","season":"Summer (Feb–May)","duration":"75–100 days","soil":"Sandy Loam","fertilizer":"DAP, Urea, MOP, Ca/B","yield":"12–25 t/ha","pests":"Aphids, Powdery Mildew","use":"Fresh fruit, juice, export","desc":"Heat-loving cucurbit requiring hot dry conditions."},
    "apple":       {"label":"Apple",        "emoji":"🍎","season":"Perennial (Jul–Oct)","duration":"150–180 days","soil":"Deep Well-drained Loam","fertilizer":"FYM, NPK, Ca, B, Mg","yield":"10–30 t/ha","pests":"Scab, Fire Blight, Codling Moth","use":"Fresh fruit, juice, cider, export","desc":"Temperate deciduous fruit requiring chilling hours."},
    "orange":      {"label":"Orange",       "emoji":"🍊","season":"Perennial (Nov–Feb)","duration":"8–12 months","soil":"Loam / Sandy Loam","fertilizer":"Urea, SSP, MOP, Zn, Fe","yield":"15–35 t/ha","pests":"Citrus psylla, Canker","use":"Fresh fruit, juice, essential oil","desc":"Subtropical citrus requiring mild winters for sugar-acid balance."},
    "papaya":      {"label":"Papaya",       "emoji":"🍈","season":"Year-round (tropical)","duration":"9–11 months","soil":"Sandy Loam","fertilizer":"FYM, NPK, Boron","yield":"40–80 t/ha","pests":"Ring spot virus, Anthracnose","use":"Fresh fruit, papain enzyme, export","desc":"Fast-growing tropical tree bearing fruit within a year."},
    "coconut":     {"label":"Coconut",      "emoji":"🥥","season":"Perennial (60–80 yr)","duration":"12 months/cohort","soil":"Coastal Sandy / Laterite","fertilizer":"Urea, Rock Phosphate, MOP","yield":"60–180 nuts/palm","pests":"Rhinoceros Beetle, Red Palm Weevil","use":"Copra, oil, water, coir","desc":"Palm of coastal tropics tolerating saline soils."},
    "cotton":      {"label":"Cotton",       "emoji":"🌿","season":"Kharif (Apr–Dec)","duration":"150–200 days","soil":"Black Cotton Soil","fertilizer":"Urea, DAP, MOP, S, Zn","yield":"1.5–4 t/ha","pests":"Bollworm, Whitefly","use":"Fiber, cottonseed oil, animal feed","desc":"Strategic fiber crop thriving in deep black soils."},
    "jute":        {"label":"Jute",         "emoji":"🌿","season":"Kharif (Mar–Jul)","duration":"100–120 days","soil":"Alluvial / Loam","fertilizer":"Urea, SSP, MOP","yield":"2–4 t/ha","pests":"Semilooper, Stem rot","use":"Fiber (bags, rope), paper pulp","desc":"Natural fiber crop requiring warm humid riverine conditions."},
    "coffee":      {"label":"Coffee",       "emoji":"☕","season":"Perennial (Oct–Feb)","duration":"8–9 months","soil":"Deep Red Laterite","fertilizer":"NPK complex, Zn, B","yield":"0.8–3 t/ha","pests":"Coffee Berry Borer, Leaf Rust","use":"Beverage, export, specialty coffee","desc":"Shade-tolerant understory crop of tropical highlands."},
}

# ── SQLite helpers ────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS soil_analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name  TEXT NOT NULL,
            N           REAL, P REAL, K REAL,
            ph          REAL, organic_matter REAL,
            notes       TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            N           REAL, P REAL, K REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL,
            top_crop    TEXT,
            all_probs   TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Routes ────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        flash("Model not loaded. Please run: python train_model.py", "danger")
        return redirect(url_for("index"))

    try:
        vals = {f: float(request.form[f]) for f in FEATURES}
    except (KeyError, ValueError) as e:
        flash(f"Invalid input: {e}", "danger")
        return redirect(url_for("index"))

    X = np.array([[vals[f] for f in FEATURES]])

    # Predicted label
    top_crop = model.predict(X)[0]

    # Probabilities for top-5
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    ranked = sorted(zip(classes, probs), key=lambda x: -x[1])[:5]

    results = []
    for crop_name, prob in ranked:
        info = CROP_INFO.get(crop_name, {})
        results.append({
            "name":    crop_name,
            "label":   info.get("label", crop_name.title()),
            "emoji":   info.get("emoji", "🌱"),
            "score":   round(prob * 100, 1),
            "season":  info.get("season", "—"),
            "yield":   info.get("yield", "—"),
            "soil":    info.get("soil", "—"),
            "fertilizer": info.get("fertilizer", "—"),
            "pests":   info.get("pests", "—"),
            "use":     info.get("use", "—"),
            "desc":    info.get("desc", ""),
        })

    # Persist to DB
    conn = get_db()
    conn.execute(
        "INSERT INTO predictions (N,P,K,temperature,humidity,ph,rainfall,top_crop,all_probs) VALUES (?,?,?,?,?,?,?,?,?)",
        (vals["N"], vals["P"], vals["K"], vals["temperature"], vals["humidity"],
         vals["ph"], vals["rainfall"], top_crop, json.dumps(ranked[:5], default=str))
    )
    conn.commit()
    conn.close()

    return render_template("result.html", results=results, inputs=vals, top_crop=top_crop)


@app.route("/crop-info")
def crop_info():
    query = request.args.get("q", "").lower()
    crops = list(CROP_INFO.items())
    if query:
        crops = [(k, v) for k, v in crops if query in k or query in v["label"].lower()]
    return render_template("crop_info.html", crops=crops, query=query)


@app.route("/crop-info/<name>")
def crop_detail(name):
    info = CROP_INFO.get(name)
    if not info:
        flash(f"Crop '{name}' not found.", "warning")
        return redirect(url_for("crop_info"))
    return render_template("crop_detail.html", name=name, info=info)


@app.route("/soil", methods=["GET", "POST"])
def soil():
    conn = get_db()
    if request.method == "POST":
        try:
            conn.execute(
                "INSERT INTO soil_analyses (field_name,N,P,K,ph,organic_matter,notes) VALUES (?,?,?,?,?,?,?)",
                (
                    request.form["field_name"],
                    float(request.form.get("N") or 0),
                    float(request.form.get("P") or 0),
                    float(request.form.get("K") or 0),
                    float(request.form.get("ph") or 7),
                    float(request.form.get("organic_matter") or 0),
                    request.form.get("notes", ""),
                )
            )
            conn.commit()
            flash("Soil analysis saved successfully!", "success")
        except Exception as e:
            flash(f"Error saving: {e}", "danger")
        conn.close()
        return redirect(url_for("soil"))

    analyses = conn.execute(
        "SELECT * FROM soil_analyses ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return render_template("soil.html", analyses=analyses)


@app.route("/soil/<int:record_id>/delete", methods=["POST"])
def delete_soil(record_id):
    conn = get_db()
    conn.execute("DELETE FROM soil_analyses WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    flash("Record deleted.", "info")
    return redirect(url_for("soil"))


@app.route("/history")
def history():
    conn = get_db()
    preds = conn.execute(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return render_template("history.html", predictions=preds)


# ── Start ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
