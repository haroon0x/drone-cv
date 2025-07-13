import os
import requests
from zipfile import ZipFile

url = 'https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-DET-train.zip'
out_dir = 'data/VisDrone2019'
os.makedirs(out_dir, exist_ok=True)
zip_path = os.path.join(out_dir, 'VisDrone2019-DET-train.zip')
if not os.path.exists(zip_path):
    r = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
with ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(out_dir) 