from ultralytics import YOLO
model = YOLO('runs/train/exp/weights/best.pt')
metrics = model.val(data='data/yolo/data.yaml')
print(metrics) 