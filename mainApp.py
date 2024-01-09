import cv2
import random
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.plots import plot_one_box
from utils.general import check_img_size, non_max_suppression, scale_coords
import time


class YOLOv7:
    def __init__(self, weights: str, image_size:int, device:str):
        self.device = device
        self.weights = weights
        self.model = attempt_load(self.weights, map_location=self.device) # Model Load FP32
        self.stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, self.stride)

        if self.device != 'cpu':
            self.half = True
        else:
            self.half = False

        if self.half:
            self.model.half() # FP16
            
        self.names = self.model.module.names if hasattr(self.model , 'module') else self.model.names
        color_values = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.colors = {i:color_values[i] for i in range(len(self.names))}

    def detect(self, raw_image: np.ndarray, conf_thresh =0.4, iou_thresh =0.3):
        # Run inference
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))
        with torch.no_grad():
            image = letterbox(raw_image, self.image_size, stride=self.stride)[0]
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = np.ascontiguousarray(image)
            image = torch.from_numpy(image).to(self.device)
            image = image.half() if self.half else image.float()
            image /= 255.0
            if image.ndimension() == 3:
                image = image.unsqueeze(0)
            # Inference
            detections = self.model(image, augment=False)[0]
            # Apply NMS
            detections = non_max_suppression(detections, conf_thresh, iou_thresh, classes=None, agnostic=False)[0]
               # Rescale boxes from img_size to raw image size
            detections[:, :4] = scale_coords(image.shape[2:], detections[:, :4], raw_image.shape).round()
            return detections

    def draw_bbox(self, img_raw:np.ndarray, predictions:torch.Tensor):
        try:
            for *xyxy, conf, cls in predictions:
                label = '%s %.2f' % (self.names[int(cls)], conf)                            
                plot_one_box(xyxy, img_raw, label=label, color=self.colors[int(cls)], line_thickness=2)
        except AttributeError:
            print("failed")
        return img_raw




if __name__=='__main__':

    yolov7=YOLOv7(weights='yolov7.pt', device='cuda', image_size=800)


    cap = cv2.VideoCapture("UAV-surveillance.mp4")   

    while cap.isOpened():
        t1 = time.time()
        read, frame = cap.read()
        if not read:
            print('A frame could not be read while video was being played.')
            break
        
        filter_categories = {
            1: 'bicycle',
            3: 'motorcycle',
            2: 'car',
            5: 'bus',
            7: 'truck'
        }

        # create a empty list for each category
        cleared_detections = {categ: [] for categ in filter_categories}

        detections = yolov7.detect(frame)
        # detection holds a liste with a tamplate like: 
        # [x, y, w, h, conf, class]
        # x and y are upper left corner coordinates
        # w and h are weight and height
        # conf is confidence of detection estimation
        # class is detection class of object

        for det in detections:
            categ = int(det[5]) # det[5] is class of object
            if categ in filter_categories: 
                cleared_detections[categ].append(det)

        # draw bounding box for each category
        for categ in cleared_detections:
            yolov7.draw_bbox(frame, cleared_detections[categ])

        # calculate number of vehicles 
        number_of_vehicles = {categ: len(cleared_detections[categ]) for categ in filter_categories}


        # DRAWING THE RECTANGLE BEHIND TEXTS
        x,y,w,h = 10, 0, 335, 125
        # the rectangle (x,y,weight,height) values
        
        rectangle = cv2.rectangle(
            frame.copy(), 
            (x,y),
            (x+w, y+h),
            (0,0,0), -1 
            # (0,0,0) means rgb color 
            # -1 means fill the rectange
        )

        transparency = 0.5
        frame = cv2.addWeighted(
            rectangle, # the rectange
            transparency,
            frame,
            1 - transparency,
            0
        )

        colors = yolov7.colors
        
        text_index = 0
        for categ_key in list(filter_categories.keys()):
            categ_value = filter_categories[categ_key]

            cv2.putText( 
                frame,
                f'Number of {categ_value}: {number_of_vehicles[categ_key]}',
                (10, 20+25*text_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, # font scale
                colors[categ_key], # color
                2 # thickness
            )
            text_index += 1

        # color_of_bicycle_bbox = yolov7.colors[1]
        # color_of_motorcycle_bbox = yolov7.colors[3]
        # color_of_car_bbox = yolov7.colors[2]
        # color_of_bus_bbox = yolov7.colors[5]
        # color_of_truck_bbox = yolov7.colors[7]


        #out.write(processed_frame)
        
        fps = int(1/(time.time() - t1))
        cv2.putText(frame, 'FPS: {}'.format(fps), (900, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        cv2.imshow('UAV YOLOv7 Surveillance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()