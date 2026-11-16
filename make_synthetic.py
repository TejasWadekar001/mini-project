"""Create a tiny fake PlantVillage-style dataset to smoke-test the pipeline.
   python make_synthetic.py synthetic_data
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_data")
random.seed(0)
CLASSES = {"Tomato___healthy": 0, "Tomato___Early_blight": 6, "Tomato___Leaf_mold": 14,
           "Potato___healthy": 0, "Potato___Early_blight": 6}

for name, n_spots in CLASSES.items():
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(40):
        g = random.randint(90, 210)
        im = Image.new("RGB", (256, 256), (g, g, g))              # grey lab background
        dr = ImageDraw.Draw(im)
        green = (random.randint(30, 70), random.randint(110, 160), random.randint(30, 70))
        dr.ellipse([30, 50, 226, 206], fill=green)
        for _ in range(n_spots):
            x, y, r = random.randint(60, 190), random.randint(75, 180), random.randint(4, 10)
            dr.ellipse([x - r, y - r, x + r, y + r], fill=(110, 70, 30))
        im.save(d / f"{i}.jpg")
print("created", out)
