#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from segment import segment
from sklearn.ensemble import AdaBoostClassifier
from segment import segment
from make_feature import make_feature
import pickle
import time

with open('adaboost_box_model.pkl', 'rb') as model_file:
	box_adaboost = pickle.load(model_file)

marker_pub = rospy.Publisher("Detection_MarkerArray", MarkerArray, queue_size=1)
marker = Marker()
markerArray_pub = None

	
prev_y_pred = None
current_y_pred = None
probability = None

def init_marker(x=0, y=0, z=0):
	marker.header.frame_id = "laser_link"
#	marker.header.stamp = rospy.Time.from_sec(time.time())
	marker.header.stamp = rospy.Time.now()
		
	marker.ns = "Detected_box"
	marker.id = 0
	marker.type = Marker.CUBE

	marker.action = Marker.ADD

	marker.pose.position.x = x
	marker.pose.position.y = y
	marker.pose.position.z = z
	marker.pose.orientation.x = 0.0
	marker.pose.orientation.y = 0.0
	marker.pose.orientation.z = 0.0
	marker.pose.orientation.w = 1.0

	marker.scale.x = 0.2
	marker.scale.y = 0.2
	marker.scale.z = 0.2

	marker.color.r = 1.0
	marker.color.g = 0.0
	marker.color.b = 0.0
	marker.color.a = 1.0

	marker.lifetime = rospy.Duration(0.15)

#marker_template = init_marker()

def scan_callback(scan):
	global prev_y_pred, current_y_pred, probability
	global marker_pub, markerArray_pub
	
	marker_array = MarkerArray()

	rows_per_seconds = 720
	data = np.zeros((rows_per_seconds, 2))

	for i in range(rows_per_seconds):
		if scan.ranges[i] < 1.0:
			data[i, 0] = scan.ranges[i] * np.cos(scan.angle_min + scan.angle_increment * i)
			data[i, 1] = scan.ranges[i] * np.sin(scan.angle_min + scan.angle_increment * i)

	Seg, Si_n, S_n = segment(data[:, :])
    
	test_X = []
	for i in range(S_n):
		seg = Seg[i][:Si_n[i]]
		feature = make_feature(data[seg, :])
		test_X += [feature]
        
	test_X = np.array(test_X)
	y_pred = box_adaboost.predict(test_X)

	observation = prev_y_pred
	state_value = current_y_pred
	
	T = np.array([[0.5, 0.4], [0.5, 0.6]])
	X = np.array([0.5, 0.5])
	
	detected = False

	#for i, pred in enumerate(y_pred):
	for i in range(len(y_pred)):
		x1 = np.dot(T, X)
		if y_pred[i] == 1:
			P = (1/(0.99*x1[0]+0.009*x1[1]))*np.dot(np.array([[0.99, 0], [0, 0.009]]), x1)
		elif y_pred[i] == 0:
			P = (1/(0.01*x1[0]+0.991*x1[1]))*np.dot(np.array([[0.01, 0], [0, 0.991]]), x1)
		X = P
		probability = X[0]
			
		if y_pred[i] == 1:
			detected = True
			seg = Seg[i][:Si_n[i]]  # 获取当前分段的数据索引
			if seg:  # 确保 seg 不为空
				x = data[Seg[i][0], 0]
				y = data[Seg[i][0], 1]
				
				marker.pose.position.x =x
				marker.pose.position.y = y
				marker.ns = "Detected_box"
				marker.type = Marker.CUBE
				marker.header.stamp = scan.header.stamp
                		
				marker_array.markers.append(marker)
                 		
			index_box = 1
			print(f"(obj_x={data[Seg[i][0], 0]}, obj_j={data[Seg[i][0], 1]}, {probability}, {index_box})")
				
	if not detected:
		print("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
	
	marker_pub.publish(marker_array)


def main():
	global markerArray_pub
	rospy.init_node("Detection_Nodes")

	rospy.Subscriber("/scan", LaserScan, scan_callback)
	
	init_marker()
	
	rospy.spin()
	


if __name__ == "__main__":
	X=np.array([0.5,0.5])
	main()
	
