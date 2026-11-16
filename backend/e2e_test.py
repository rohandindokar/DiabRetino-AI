"""Quick end-to-end check of the running backend on the 5 reference samples."""
import json
import requests

base = "http://127.0.0.1:8000"
samples = [
    ("stage_0_no_dr.png", 0),
    ("stage_1_mild.png", 1),
    ("stage_2_moderate.png", 2),
    ("stage_3_severe.png", 3),
    ("stage_4_proliferative.png", 4),
]

h = requests.get(base + "/api/health").json()
print("health: device=%s  flagship ckpt=%s"
      % (h["device"], h["checkpoints_loaded"]["hybrid_vit_cnn"]))
print()

correct = 0
for fname, expected in samples:
    r = requests.post(base + "/api/predict",
                      data={"sample_filename": fname, "model_name": "hybrid_vit_cnn"})
    if r.status_code != 200:
        print(fname, "-> HTTP", r.status_code, r.text[:300])
        continue
    d = r.json()
    p = d.get("prediction", {})
    hit = "Y" if p.get("class_id") == expected else "n"
    correct += (p.get("class_id") == expected)
    print("%s  %-26s  pred=%s (%s)  exp=%d  conf=%s%%  laser=%s"
          % (hit, fname, p.get("class_id"), p.get("class_name"), expected,
             p.get("confidence_percentage"), p.get("laser_required")))

try:
    m = requests.get(base + "/api/metrics").json()
    print("\n/api/metrics OK  models:", [x["model"] for x in m["models"]])
except Exception as e:
    print("\n/api/metrics not ready:", e)

d0 = requests.post(base + "/api/predict",
                   data={"sample_filename": "stage_4_proliferative.png",
                         "model_name": "hybrid_vit_cnn"}).json()
if "visualizations" in d0:
    print("\nheatmap present:", bool(d0["visualizations"]["heatmap"]))
    print("overlay present:", bool(d0["visualizations"]["overlay"]))
print("\nReference samples matched expected stage: %d/5" % correct)
