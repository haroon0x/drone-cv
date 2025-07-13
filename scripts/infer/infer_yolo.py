from ultralytics import YOLO
import os
glob = os.path.join('data/yolo/visdrone/images', '*')
model = YOLO('runs/train/exp/weights/best.pt')
model.predict(source=glob, project='runs/infer', name='exp', save=True) 