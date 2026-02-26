from ultralytics import YOLO

model = YOLO("../yolov11models/yolo11s-seg.pt")
reults = model.train(data="D:/Projects/01_DermaIQ/woundSegmentation/data/wound-seg.yaml", epochs=200, imgsz=640, workers=0)