#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math

STATE_COUNT_THRESHOLD = 2 # 3#2

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        
        # Initialize Variables
        # Variables for the callback - pose_cb
        self.pose = None
        # Variables for the callback - waypoints_cb
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        # Variables for the callback - image_cb
        self.camera_image = None
        self.using_camera = True
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        # Variables for the callback - traffic_cb
        self.lights = []

        # Get the (x,y) parametrs of the stop line
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        # Get the is_site boolean from the self.config
        #is_site = self.config['is_site']
        
        
        # Variables for classifier
        self.bridge = CvBridge()
        # send the path of the respective graph to the classifier
        #self.is_site = self.config['is_site']
        rospy.loginfo("Is site mode on? : ")
        rospy.loginfo(self.config['is_site'])
        
        self.light_classifier = TLClassifier(self.config['is_site'])
        
        self.listener = tf.TransformListener()
        
        # Chcek is camera is on
        #if self.using_camera:
         #   rospy.loginfo("Using camera")
        #else:
         #   rospy.loginfo("Not using Camera")


        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
            
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        rospy.spin()

    # call back function for current_pose topic (rospy.subscriber)
    # This subscriber will be called frequently as it is not latched and we need position of car at every step.
    def pose_cb(self, msg):
        # TODO: Implement
        # 1. Get the current position of Carla
        self.pose = msg

    # call back function for base_waypoints topic (rospy.subscriber)
    # This is a latched subscriber thats why this is done once as the base waypoints are never changing.
    def waypoints_cb(self, waypoints):
        # TODO: Implement
        # 1. Here we are basically storing the waypoints in the object self.waypoints
        self.waypoints = waypoints
        # 2. Here we are using KDTree which helps to lookup points closest to the Carla.
        if not self.waypoints_2d:
            # 3. Converting the waypoints to 2D coordinates x and y (look at the provided structure for message type)
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # 4. Implement the KDtree
            self.waypoint_tree = KDTree(self.waypoints_2d)
            
    # call back function for /vehicle/traffic_lights topic (rospy.subscriber)
    # This provides you with the location of the traffic light in 3D map space and helps you acquire an accurate ground truth data source for the traffic           light classifier by sending the current color state of all traffic lights in the simulator.
    def traffic_cb(self, msg):
        self.lights = msg.lights
        
    # call back function for /image_color topic (rospy.subscriber)
    # This subscriber provides the camera images required to predict the traffic light and the state of light
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        #1. Turn the has_image = True
        self.has_image = True
        #2. Get the Camera Image
        self.camera_image = msg
        #3. Call the function and get the stop light waypoint and also the state of the light
        light_wp, state = self.process_traffic_lights()
        
        # Log info of the state detected
        if state == TrafficLight.RED:
            rospy.loginfo("Detected RED light")
        elif state == TrafficLight.YELLOW:
            rospy.loginfo("Detected YELLOW light")
        elif state == TrafficLight.GREEN:
            rospy.loginfo("Detected GREEN light")
        else:
            rospy.loginfo("No traffic light detected")

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        # 4. If new state of light is not same as state
        if self.state != state:
            # set counter to 0
            self.state_count = 0
            # update self state to new state
            self.state = state
        # 5. If the state if still not changed and counter has reached the max_counter_set = 3
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            # Update the last_state as the self.state
            self.last_state = self.state
            # if the state of light is red/yellow then update light_wp with the stop light waypoint else -1 (so Carla wont stop at the traffic light if not                 RED)
            if state == TrafficLight.RED or state == TrafficLight.YELLOW:
            #if state == TrafficLight.RED:
                light_wp = light_wp 
            else:
                light_wp = -1
            #light_wp = light_wp if state == TrafficLight.RED else -1
            
            # Update the last_waypoint as the self.last_wp
            self.last_wp = light_wp
            # publish the upcoming red light topic with the light_wp value -1 or light_wp dpends on the state
            self.upcoming_red_light_pub.publish(Int32(light_wp))
            
        # 6 the state of light might change so we need to keep a track of it
        else: # publish the upcoming red light topic
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            
        # increment the counter
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        #1. Get the closest waypoint
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # 3. Check if closest is ahead or behind vehicle - we want is in front/ahead and not behind
        #closest_coord = self.waypoints_2d[closest_idx]
        #prev_coord = self.waypoints_2d[closest_idx - 1]

        # 4. Equation for hyperplane through closest_coords
        #cl_vect = np.array(closest_coord)
        #prev_vect = np.array(prev_coord)
        #pos_vect = np.array([x, y])
        
        # 5. Take the product to see if it is in front or behind
        #val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        
        # 6. if the product is positive the it is actually behind the car, then take the next waypoint
        #if val > 0:
         #   closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx
        

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # 1. For testing in simulator
        classification = light.state
        return classification
#-------------------------------------------------------------------------------------------------------------------------------------------------        
        # 2. When in driving mode
        # 2.1 if no image is captured , return False
        #if(not self.has_image):
         #   self.prev_light_loc = None
          #  return False
        # 2.2 else take the imgae from the camera topic subscriber
        #cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #2.3 Get classification of the image RED/GREEN/YELLOW
        #classification = self.light_classifier.get_classification(cv_image, self.config['is_site'])
        #return classification

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #1. Initialize the variable of the closest trraffic light and the index of the stop line to None
        closest_light = None
        line_wp_idx = None

        #2. Get List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        
        #3. Get the closest waypoint from position of the car use the KDTree for same
        if(self.pose):
            #3.1 X and Y coordianted of car position
            car_x_position = self.pose.pose.position.x
            car_y_position = self.pose.pose.position.y
            #3.2 Get the closest waypoint
            car_wp_idx = self.get_closest_waypoint(car_x_position, car_y_position)

            #TODO find the closest visible traffic light (if one exists)
            #4. Get length of total number of waypoints
            diff = len(self.waypoints.waypoints)
        
            #5. here we will loop through all the list of traffic light in our track and determine next upcoming one for Carla
            #. Loop thorugh all the traffic light list from acquired from self.lights
            for i, light in enumerate(self.lights):
                #5.1 Get each stop line position (we should be having 8 for the simulator mode)
                line = stop_line_positions[i]
                #5.2 Get the waypoint for the stop line
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                #5.3 get the difference between current car closest waypoint and stop-line closest waypoint
                d = temp_wp_idx - car_wp_idx # note should be always in front of car
                #5.4 check if difference if within the loop length
                if d>=0 and d<diff:
                    #5.4.1 if yes, get the distance d
                    diff = d
                    #5.4.2 update the closest_light index
                    closest_light = light
                    #5.4.3 update the closest stop-line index
                    line_wp_idx = temp_wp_idx
                    
        #6. If we detect the traffic light then return the index of the stop line and the state of the light
        if closest_light and diff < 150:
            state = self.get_light_state(closest_light)
            #print(state)
            print(diff)
            return line_wp_idx, state
        
        #self.waypoints = None
        
        #6. If we do not detect any light return the line stop index as -1, so that the car is not stopping and state of light as unknown
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
