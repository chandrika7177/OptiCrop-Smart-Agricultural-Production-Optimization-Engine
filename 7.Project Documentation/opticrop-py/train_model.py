"""
train_model.py
Generates synthetic crop_recommendation.csv from ideal-range profiles,
then trains a RandomForestClassifier and saves model.pkl.
Run once before starting the app:  python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# ------------------------------------------------------------------
# Crop profiles:  label, N, P, K, temperature, humidity, ph, rainfall
# Each tuple: (label, N_min, N_max, P_min, P_max, K_min, K_max,
#              temp_min, temp_max, hum_min, hum_max, ph_min, ph_max,
#              rain_min, rain_max)
# ------------------------------------------------------------------
PROFILES = [
    ("rice",         60,120, 30,60,  30,60,  20,28, 80,90, 5.5,7.0, 150,300),
    ("maize",        60,120, 40,70,  30,60,  18,30, 55,80, 5.5,7.5,  60,120),
    ("chickpea",     20,50,  40,80,  40,80,  15,25, 14,28, 5.5,7.5,  60,105),
    ("kidneybeans",  15,35,  50,80,  15,35,  15,27, 20,40, 5.5,7.0,  70,120),
    ("pigeonpeas",   15,35,  50,70,  15,35,  25,37, 40,70, 5.0,7.5,  95,210),
    ("mothbeans",    15,30,  35,60,  30,50,  24,38, 40,70, 3.5,6.5,  35,70),
    ("mungbean",     15,30,  35,70,  15,40,  25,36, 75,95, 6.2,7.2,  50,75),
    ("blackgram",    15,30,  35,65,  15,35,  25,37, 60,85, 5.5,7.0,  55,80),
    ("lentil",       10,30,  50,70,  15,30,  10,22, 60,80, 6.5,7.5,  35,55),
    ("pomegranate",  15,30,  15,40,  15,40,  18,36, 85,96, 5.5,7.5, 105,130),
    ("banana",       80,140, 60,85,  40,65,  24,32, 75,92, 5.5,6.5, 100,180),
    ("mango",        15,30,  10,20,  10,30,  24,32, 40,65, 4.5,6.5,  85,130),
    ("grapes",       15,30,  10,25,  15,30,   8,24, 78,92, 5.5,6.5,  60,85),
    ("watermelon",   80,130, 40,65,  40,65,  24,38, 80,95, 5.5,7.0,  45,80),
    ("muskmelon",    80,120, 40,60,  40,60,  28,40, 88,96, 6.0,7.5,  20,40),
    ("apple",         0,25,  90,145,140,210,   7,20, 86,96, 5.5,6.5, 105,130),
    ("orange",       10,25,   5,15,   5,15,  10,25, 86,96, 6.0,7.5, 105,130),
    ("papaya",       40,65,  25,55,  40,70,  24,36, 86,96, 6.0,7.0, 150,180),
    ("coconut",       3,20,   5,25,  10,30,  25,34, 85,96, 5.0,8.0, 145,210),
    ("cotton",       95,140, 25,55,  15,35,  20,32, 50,70, 6.0,8.0,  60,100),
    ("jute",         55,95,  25,60,  25,65,  24,37, 65,92, 6.0,7.0, 145,230),
    ("coffee",       75,130, 25,55,  20,55,  18,30, 50,70, 6.0,6.5, 150,180),
]

SAMPLES_PER_CROP = 120
RNG = np.random.default_rng(42)


def generate_dataset():
    rows = []
    for p in PROFILES:
        label = p[0]
        N_lo, N_hi   = p[1],  p[2]
        P_lo, P_hi   = p[3],  p[4]
        K_lo, K_hi   = p[5],  p[6]
        t_lo, t_hi   = p[7],  p[8]
        h_lo, h_hi   = p[9],  p[10]
        ph_lo, ph_hi = p[11], p[12]
        r_lo, r_hi   = p[13], p[14]

        n  = RNG.uniform(N_lo,  N_hi,  SAMPLES_PER_CROP)
        pp = RNG.uniform(P_lo,  P_hi,  SAMPLES_PER_CROP)
        k  = RNG.uniform(K_lo,  K_hi,  SAMPLES_PER_CROP)
        t  = RNG.uniform(t_lo,  t_hi,  SAMPLES_PER_CROP)
        h  = RNG.uniform(h_lo,  h_hi,  SAMPLES_PER_CROP)
        ph = RNG.uniform(ph_lo, ph_hi, SAMPLES_PER_CROP)
        r  = RNG.uniform(r_lo,  r_hi,  SAMPLES_PER_CROP)

        for i in range(SAMPLES_PER_CROP):
            rows.append({
                "N":           round(float(n[i]),  2),
                "P":           round(float(pp[i]), 2),
                "K":           round(float(k[i]),  2),
                "temperature": round(float(t[i]),  2),
                "humidity":    round(float(h[i]),  2),
                "ph":          round(float(ph[i]), 2),
                "rainfall":    round(float(r[i]),  2),
                "label":       label,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(os.path.dirname(__file__), "crop_recommendation.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅  Saved {len(df)} rows → {csv_path}")
    return df


def train(df: pd.DataFrame):
    FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    X = df[FEATURES].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"✅  Accuracy on hold-out set: {acc * 100:.1f}%")

    pkl_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(clf, pkl_path)
    print(f"✅  Model saved → {pkl_path}")
    return clf


if __name__ == "__main__":
    print("🌱  Generating dataset …")
    df = generate_dataset()
    print("🤖  Training RandomForestClassifier …")
    train(df)
    print("🎉  Done!")
