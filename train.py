from ultralytics import YOLO

if __name__ ==  "__main__":
    model = YOLO(r'D:\DRIVE\TUP Files\4A\Thesis\model\yolo\results\train\weights\last.pt')
    model.train(resume=True)
    #results = model.train(data=r'D:\DRIVE\TUP Files\4A\Thesis\model\yolo\data.yaml', epochs=80, imgsz=640, project=r'D:\DRIVE\TUP Files\4A\Thesis\model\yolo\results', device=0, workers=0, optimize= True, lr0=0.001, lrf=0.01, fliplr=0.0, warmup_epochs = 0,batch=4 ,amp=False)