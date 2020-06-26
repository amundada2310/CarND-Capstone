from styx_msgs.msg import TrafficLight
import numpy as np
import tensorflow as tf
import os
import cv2
import rospy
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        
        # 1, check if the is_site bool is on or off
        if is_site: # if site mode on use the below pre-trained graph
            model_path = r'./light_classification/models/ssd_real/frozen_inference_graph.pb'
            
        else:#2 if off use the below pre-trained graph
            model_path = r'./light_classification/models/ssd_sim/frozen_inference_graph.pb'
            
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')
        
        # 3. Get session
        self.sess = tf.Session(graph=self.graph)  
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    

    def get_classification(self, image, is_site):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        #TODO implement light color prediction
        
        #1. dstack the incoming image
        #image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
            
        #2. convert the image in array form    
        #image_expanded = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        
        
        with self.graph.as_default(): 

            
            #(im_width, im_height) = image.size
            #image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
            #image_expanded = np.expand_dims(image_np, axis=0)
            image_expanded = np.expand_dims(image, axis=0)
            
            #t0 = datetime.datetime.now()
            
            (boxes, scores, classes, num) = self.sess.run([self.detect_boxes, self.detect_scores, self.detect_classes, self.num_detections],
                                                     feed_dict={self.image_tensor: image_expanded})
            
            #t1 = datetime.datetime.now()
            
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            
        # set the minimum confidence cutoff
        confidence_cutoff = 0.20#0.80#0.50
        # only take the class with max(score) > confidence_cutoff

        if scores[0] > confidence_cutoff:
            # print name of the identified class, associated scores and time taken to identify and return the class
            if classes[0] == 1:
                print('GREEN') 
                print(scores[0])
                #time_final = (t1 - t0)*1000 
                #print("Time ", time_final, "\n")
                return TrafficLight.GREEN
                    
            elif classes[0] == 2:
                print('RED')
                print(scores[0])
                #time_final = (t1 - t0)*1000
                #print("Time ", time_final, "\n")
                return TrafficLight.RED
                    
            elif classes[0] == 3:
                print('YELLOW')
                print(scores[0])
                #time_final = (t1 - t0)*1000
                #print("Time ", time_final, "\n")
                return TrafficLight.YELLOW
        
        return TrafficLight.UNKNOWN