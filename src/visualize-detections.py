import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree
import cv2
from random import randint

def get_scored_data(scored_dir):
	annotation_files = [f for f in os.listdir(scored_dir) if (os.path.isfile(os.path.join(scored_dir, f)) and not f == ".DS_Store")]
	ground_truth = {}
	for annotation_file in annotation_files:
		base_file, _ = os.path.splitext(annotation_file)
		base_file, _ = os.path.splitext(base_file)
		faces = []

		with open(os.path.join(scored_dir, annotation_file)) as open_file:
			annotations = json.load(open_file)

		for face_data in annotations['faces']:
			face = [face_data['xmin'],face_data['ymin'],face_data['xmax'],face_data['ymax']]
			faces.append(face)
		ground_truth[base_file] = faces
	return ground_truth

def get_ground_truth(annotation_dir):
	annotation_files = [f for f in os.listdir(annotation_dir) if (os.path.isfile(os.path.join(annotation_dir, f)) and not f == ".DS_Store")]
	ground_truth = {}
	for annotation_file in annotation_files:
		base_file, ext = os.path.splitext(annotation_file)
		faces = []
		e = xml.etree.ElementTree.parse(os.path.join(annotation_dir, annotation_file)).getroot()
		for b in e.findall('.//object/bndbox'):
			face = [int(b.find("xmin").text), int(b.find("ymin").text), int(b.find("xmax").text), int(b.find("ymax").text)]
			faces.append(face)
		ground_truth[base_file] = faces
	return ground_truth

def score_detections():

	# overlap_threshold = 0.5
	dir_path = os.path.dirname(os.path.realpath(__file__))
	image_dir = os.path.join(dir_path,"..","images")
	annotation_dir = os.path.join(dir_path,"..","annotations")
	results_dir = os.path.join(dir_path,"..","results")
	visualization_dir = os.path.join(dir_path,"..","visualization")

	print 'Reading in ground truth data...'
	ground_truth_data = get_ground_truth(annotation_dir)
	result_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
	scored_datas = {d:get_scored_data(os.path.join(results_dir, d)) for d in result_dirs}

	df = pd.DataFrame(scored_datas.keys(),columns=['name'])
	
	overlap_threshold = 0.5

	filenames = scored_datas[scored_datas.keys()[0]]

	results = []

	colors = {}
	for scored_data_name in scored_datas:
		colors[scored_data_name] = (randint(0, 255), randint(0, 255), randint(0, 255))

	font = cv2.FONT_HERSHEY_SIMPLEX

	for file_name in filenames:
		image_path = os.path.join(image_dir, file_name + ".jpg")
		image = cv2.imread(image_path)
		height, width, channels = image.shape

		start = 10
		for scored_data_name in scored_datas:
			cv2.rectangle(image, (10, start), (30, start+20), colors[scored_data_name], -1)
			cv2.putText(image,scored_data_name,(40,start+17), font, 0.75,colors[scored_data_name],2,cv2.LINE_AA)
			start += 30

		for box in ground_truth_data[file_name]:
			cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), 4)
		for scored_data_name in scored_datas:
			# print scored_data_name
			predicted_list = scored_datas[scored_data_name][file_name]
			ground_truth = ground_truth_data[file_name]
			
			for box in predicted_list:
				cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colors[scored_data_name], 3)
		# cv2.imshow("Faces found" ,image)
		cv2.imwrite(os.path.join(visualization_dir, file_name + ".jpg"), image)
		# cv2.waitKey(0)
			# print 'AP: [%f] AR: [%f]' % (average_precision, average_recall)

if __name__ == "__main__":
	score_detections()