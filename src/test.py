import os
import json
import numpy as np
import pandas as pd
import xml.etree.ElementTree

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

	print 'Reading in ground truth data...'
	ground_truth_data = get_ground_truth(annotation_dir)
	
	faces_count = 0
	max_faces = -1
	for g in ground_truth_data:
		faces_count += len(ground_truth_data[g])
		if len(ground_truth_data[g]) > max_faces:
			max_faces = len(ground_truth_data[g])

	artists = list(set([a.split("-", -1)[0] for a in ground_truth_data.keys()]))
	total_images = len(ground_truth_data)

	print faces_count
	number_artists = len(artists)

	print total_images
	print number_artists

	print float(faces_count) / total_images
	print float(faces_count) / number_artists

	print max_faces

	# df = pd.DataFrame(scored_datas.keys(),columns=['name'])
	# print df
	# for overlap_threshold in [0.9,0.7,0.5,0.3,0.1,0.01]:
	# 	# print overlap_threshold
	# 	results = []
	# 	for scored_data_name in scored_datas:
	# 		# print scored_data_name
	# 		average_precision = 0.0
	# 		average_recall = 0.0
	# 		scored_data = scored_datas[scored_data_name]
	# 		for file_name in scored_data:
	# 			predicted_list = scored_data[file_name]
	# 			ground_truth = ground_truth_data[file_name]
	# 			matches = np.zeros((len(ground_truth)))
	# 			false_negative = 0
	# 			false_postive = 0
	# 			true_positive = 0
	# 			for p in predicted_list:
	# 				best_overlap = -1000
	# 				best_match = None
	# 				for g_i, g in enumerate(ground_truth):
	# 					current_overlap  = bb_intersection_over_union(p, g)
	# 					if current_overlap > best_overlap:
	# 						best_overlap = current_overlap
	# 						best_match = g_i
	# 				if not best_match is None and best_overlap >= overlap_threshold:
	# 					matches[best_match] += 1
	# 				else:
	# 					false_postive += 1

	# 			for g in matches:
	# 				if g > 0:
	# 					true_positive += 1
	# 					false_postive += int(g)-1
	# 				else:
	# 					false_negative += 1

	# 			precision = 0.0
	# 			recall = 0.0
	# 			if (true_positive > 0 or false_postive > 0):
	# 				precision = float(true_positive) / (true_positive + false_postive)
	# 			if (true_positive > 0 or false_negative > 0):
	# 				recall = float(true_positive) / (true_positive + false_negative)
	# 			average_precision += precision
	# 			average_recall += recall
	# 			# print 'TP: [%d] FP:[%d] FN: [%d]' % (true_positive, false_postive, false_negative)
	# 			# print 'P: [%f] R: [%f]' % (precision, recall)
	# 			# print predicted_list
	# 			# print ground_truth
	# 		average_precision /= len(scored_data)
	# 		average_recall /= len(scored_data)
	# 		f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall)
	# 		results.append(f1)
	# 	df[overlap_threshold] = results
	# 		# print 'AP: [%f] AR: [%f]' % (average_precision, average_recall)

	# print df
	# df.to_csv(os.path.join(dir_path,'..','results.csv'))
	# print scored_data



if __name__ == "__main__":
	score_detections()