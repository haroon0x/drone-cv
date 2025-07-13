import os
import shutil
import random
img_dir = 'data/yolo/visdrone/images'
lbl_dir = 'data/yolo/visdrone/labels'
train_img_dir = 'data/yolo/visdrone/train/images'
train_lbl_dir = 'data/yolo/visdrone/train/labels'
val_img_dir = 'data/yolo/visdrone/val/images'
val_lbl_dir = 'data/yolo/visdrone/val/labels'
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)
imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(imgs)
split = int(0.8 * len(imgs))
train_imgs = imgs[:split]
val_imgs = imgs[split:]
for f in train_imgs:
    shutil.copy(os.path.join(img_dir, f), os.path.join(train_img_dir, f))
    lbl = os.path.splitext(f)[0] + '.txt'
    if os.path.exists(os.path.join(lbl_dir, lbl)):
        shutil.copy(os.path.join(lbl_dir, lbl), os.path.join(train_lbl_dir, lbl))
for f in val_imgs:
    shutil.copy(os.path.join(img_dir, f), os.path.join(val_img_dir, f))
    lbl = os.path.splitext(f)[0] + '.txt'
    if os.path.exists(os.path.join(lbl_dir, lbl)):
        shutil.copy(os.path.join(lbl_dir, lbl), os.path.join(val_lbl_dir, lbl)) 