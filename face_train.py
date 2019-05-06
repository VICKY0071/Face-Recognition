import os
from pillow import Image
import numpy as np

BASE_DIR = os.path.dirnname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

for files, root, dirs in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = path.os.join(root, file)
			label = os.path.basename(root).replace(' ', '-').lower()
			print(label, path)
			pil_image = Image.open(path).convert('L') # L is for grayscale
			image_array = np.array(pil_image, "uint8")
			