
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\fog_11\ultralytics_niou\ultralytics\cfg\models\11+2\yolo11_DSConv+ese+orepa.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'D:\fog_11\ultralytics_niou\ultralytics\cfg\datasets\my_detect_wu.yaml',
                cache=False,
                imgsz=640,
                epochs=600,
                batch=64,
                close_mosaic=0,
                workers=16,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=True, # close amp
                # fraction=0.2,
                lr0=0.01,
                lrf=0.01,
                project='runs/duibi',
                name='ours',
                )
    # data=r'ultralytics/cfg/datasets/my_detect.yaml',