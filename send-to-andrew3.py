from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import json

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class eventsHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            
            time.sleep(0.1) # test sleep tenth of a second
            # run detect
            with torch.no_grad():
                detect(event.src_path, "FinalFinalBest.pt")
                
def detect(source, weights):

    start_time = time.time()

    save_img = True
    imgsz = 640

    # Directories
    save_dir = Path(increment_path(Path('runs/detect') / "detection", exist_ok = False))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        print("2nd stage")

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
            print("Classified")

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                file_path = str(save_dir / "Labels.json")
                # file_path = "example.json"

                with open(file_path, 'a') as json_file:
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        jsonData = { "Name": names[int(c)]}
                        json.dump(jsonData, json_file)
                        json_file.write('\n')

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    # print(f" The image with the result is saved in: {save_path}")

                else:
                    print("ERROR: not given image")

    if save_img:
        s = ''

    print("Detect Time: " + (time.time() - start_time))


if __name__ == "__main__":

    start_time = time.time()
    # Set up fusing model layers
    # Initialize
    set_logging()
    device = select_device('')

    # Load model
    model = attempt_load("FinalFinalBest.pt", map_location=device)  # load FP32 model, check map_location if does not work
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(640, s=stride)  # check img_size

    print("Model Load Time: " + (time.time() - start_time))

    # do stuff
    path = "/home/Guardian-Eye/Pictures/Events/" # try with different paths
    event_handler = eventsHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        
    start_time = time.time()
    observer.join()
    print("observer Time: " + time.time() - start_time)
    

