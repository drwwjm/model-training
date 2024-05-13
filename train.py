from ultralytics import YOLO

if __name__ ==  "__main__":
    model = YOLO('yolov8x.pt')

    results = model.train(data=r'D:\DRIVE\TUP Files\4A\Thesis\model\yolo\data.yaml', epochs=100, imgsz=640, project=r'D:\DRIVE\TUP Files\4A\Thesis\model\yolo\results', device=0, workers=0, optimize= True, lr0=0.001, lrf=0.01, fliplr=0.0,amp=False, batch=2)