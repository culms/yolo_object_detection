import cv2
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import keras
import sys
import os

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)




yolo_model =load_model("yolo.h5")



yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))






def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence."""
    
   
    box_scores = box_confidence * box_class_probs
    
   
    box_classes = keras.backend.argmax(box_scores, axis=-1)
    box_class_scores = keras.backend.max(box_scores, axis=-1 )
    
    # Create a filtering mask based on "box_class_scores" by using "threshold". 
    filtering_mask = box_class_scores >= threshold
   
    #  Apply the mask to scores, boxes and classes
    
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes =  tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes


def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = np.max([box1[0], box2[0]])
    yi1 = np.max([box1[1], box2[1]])
    xi2 = np.min([box1[2], box2[2]])
    yi2 = np.min([box1[3], box2[3]])
    inter_area = (yi2 - yi1) * (xi2 - xi1)
   
    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box2[3] - box2[1]) * (box2[2] - box2[0])
    union_area = box1_area + box2_area - inter_area
   
    
    
    iou = inter_area / union_area

    return iou



def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    """
   
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
   
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    
    
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    
    return scores, boxes, classes




def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    """
    
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)
    
    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes =  yolo_non_max_suppression(scores, boxes, classes)
    
    
    return scores, boxes, classes




sess = K.get_session()


box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.
    """


    # Preprocess  image
    image, image_data = preprocess_image(image_file, model_image_size = (608, 608))
    
    
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
   

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
   
    colors = generate_colors(class_names)
   
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
   
    image.save(os.path.join("out", image_file), quality=90)
    # Display the results in the notebook
    #t= plt.imshow(image)
    #print(image)
    #plt.show()
    
    
    return out_scores, out_boxes, out_classes



def video_2_frame(Video_addr):
    vidcap = cv2.VideoCapture(Video_addr)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        print ('Read a new frame%d: '% count, success)
        if success:    
            image = cv2.resize(image,(1280,720)) 
        
        cv2.imwrite("input/frame%d.jpg" % count, image)     # save frame as JPEG file
        count += 1
        if cv2.waitKey(10) == 27:                    # exit if Escape is hit
            break
    return count        


def frames_2_video(video_name):
    image_folder= 'out\input'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name,fourcc, 14, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()



def call_pred(count):
	for i in range(count):
		out_scores, out_boxes, out_classes = predict(sess, "input/frame%d.jpg" % i)
		print ('\n\n ',(i*100)/count ,"% complete")





def delete_trace(dir_name):
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir_name, item))
"""         
def play_video(video_name):
    cap = cv2.VideoCapture(video_name)

    while(cap.isOpened()):
        ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""


def main_call(video_name,new_name):
    count = video_2_frame(video_name)
    count = count -2
    call_pred(count)
    frames_2_video(new_name)
    path = "C:/Users/Mahipal/Downloads/project/out/input"          
    delete_trace(path)
    path = "C:/Users/Mahipal/Downloads/project/input"          
    delete_trace(path)
    #play_video()




arg1 = sys.argv[1]
arg2 = sys.argv[2]
main_call(arg1,arg2)



    
    



        


