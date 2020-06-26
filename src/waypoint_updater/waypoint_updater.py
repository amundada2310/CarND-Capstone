#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''

This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 70 # Number of waypoints we will publish. You can change this number#20#15
RATE_PUBLISHING = 30 # How frequently we wnat the data to be published (in Hz) # Here we can go as low as 20 # 20#25#30
STOP_BEFORE_WAYPOINTS =5# Number of waypoints to stop before the stop line is reached
MAX_DECEL = 1 # Maximum Deceleration is set to 0.5 #1

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        #Initialize the variables
        # Variables for the callback - waypoints_cb
        self.base_lane = None
        #self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        # Variables for the callback - pose_cb
        self.pose = None
        # Variable for callback - traffic_cb
        self.stopline_wp_idx = -1
        
        # This subscriber (topic- /current_pose) provides a current position of Carla.
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # This subscriber (topic- /base_waypoints) provides a complete list of waypoints Carla will be following.
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        #rospy.spin()
        # This self.loop is defined to get control over resulting frequency - how frequenty we want the publisher to publish the message
        self.loop()
        
        
    # Defining the rospy.loop
    def loop(self):
        # 1. select the rate here it is 50Hz
        rate = rospy.Rate(RATE_PUBLISHING)
        # 2. If the rospy is still running
        while not rospy.is_shutdown():
            # 3. And if we have the current positiona nd base waypoints from the subscribers
            if self.pose and self.base_lane:
                # 4. Call function and Get the first or closests waypoint in front of the car
                #closest_waypoint_idx = self.get_closest_waypoint_idx()
                # 5. Publish that closest waypoint index
                #self.publish_waypoints(closest_waypoint_idx)
                self.publish_waypoints()
                
            rate.sleep()
            
    # Define the get_closest_waypoint_idx function
    def get_closest_waypoint_idx(self):
        # 1. get the x and y poistion of the current pose
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        # 2. query the kdtree to get the index and position of the 1st closest waypoint
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # 3. Check if closest is ahead or behind vehicle - we want is in front/ahead and not behind
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # 4. Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        
        # 5. Take the product to see if it is in front or behind
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        
        # 6. if the product is positive the it is actually behind the car, then take the next waypoint
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx
                
    # Define the publish waypoint function 
    # This function actually porvides the list of waypoints from the closest waypoint ahead of car till the next LOOKAHEAD_WPS waypoints
    #def publish_waypoints(self, closest_idx): 
    def publish_waypoints(self):     
        # 1. Call the genearte Lane function
        final_lane = self.generate_lane()
        # 1. Create a new lane msg
        #lane = Lane()
        # 2. Get the farthest index for the waypoints
        #farthest_idx = closest_idx + LOOKAHEAD_WPS 
        # 3. Final waypoints
        #lane.waypoints =  self.base_waypoints.waypoints[closest_idx:farthest_idx]
        # 4. Publish the message of final waypoints on the /final_waypoints topic
        self.final_waypoints_pub.publish(final_lane)
        
    # Generate waypoints and update their velocity depending how we want carla to behave
    def generate_lane(self):
        # 1. Create a new lane msg
        lane = Lane()
        # 2. Call function and Get the first or closests waypoint in front of the car
        closest_idx = self.get_closest_waypoint_idx()
        # 3. Get the farthest index for the waypoints
        farthest_idx = closest_idx + LOOKAHEAD_WPS 
        # 4. Get Base Waypoints from closest to farthest
        base_waypoints = self.base_lane.waypoints[closest_idx:farthest_idx]
        # 5. check if there is red or yellow light ahead         
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farthest_idx):
            # if not, just print the base_waypoints as it is and we can drive without reducing speed
            lane.waypoints = base_waypoints
        else:# 6. but if we detect then we need to reduce our speed and get ready to stop ahead
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
            
        return lane
    
    # Define the deceleartion function if we see the RED Light
    def decelerate_waypoints(self, waypoints, closest_idx):
        # 1.create a temporary list of new waypoints
        temp = []
        #2. Loop through the list of original waypoints
        for i, wp in enumerate(waypoints):
            #3. Create new waypoint message
            p = Waypoint()
            # 4. Copy the position of the original waypoints as the positionand orientation is same only velocity at waypoints is changing
            p.pose = wp.pose
            #5. Get the stopping waypoint index for Carla
            stop_idx = max(self.stopline_wp_idx - closest_idx - STOP_BEFORE_WAYPOINTS, 0)
            # 6. Get the total distance before we stop at the line before Traffic light
            dist = self.distance(waypoints, i, stop_idx)
            # 7. Update the velocity at each waypoint
            #vel = math.sqrt(2 * MAX_DECEL * dist) + ((stop_idx - i + 1)/stop_idx)
            vel = (math.sqrt(2 * MAX_DECEL * dist)) 
            print(vel)
            # 8. Check if the velocity is very less, just return 0
            if vel < 1.0:
                vel = 0.0
            # 9. Get the minimum of 2 velocities and update velocity of that waypoint
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            #p.twist.twist.linear.x = (min(vel, wp.twist.twist.linear.x))*0.90 # reducing the velocity considering a safety precaution 
            print(p.twist.twist.linear.x)
            # 10. append the waypoit list
            temp.append(p)
            
        return temp
        
        
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
        # 1. Here we are basically storing the waypoints in the object self.base_waypoints
        self.base_lane = waypoints
        # 2. Here we are using KDTree which helps to lookup points closest to the Carla.
        if not self.waypoints_2d:
            # 3. Converting the waypoints to 2D coordinates x and y (look at the provided structure for message type)
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            # 4. Implement the KDtree
            self.waypoint_tree = KDTree(self.waypoints_2d)
        

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity
        print(waypoints[waypoint].twist.twist.linear.x)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
        
