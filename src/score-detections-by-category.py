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
	dataset_file = os.path.join(dir_path,"..","dataset.csv")

	print 'Reading in ground truth data...'
	ground_truth_data = get_ground_truth(annotation_dir)
	result_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
	scored_datas = {d:get_scored_data(os.path.join(results_dir, d)) for d in result_dirs}

	dataset_df = pd.read_csv(dataset_file)

	df = pd.DataFrame(scored_datas.keys(),columns=['name'])
	print df

	category_to_filenames = {}

	for _, row in dataset_df.iterrows():
		category = row['category']
		if category not in category_to_filenames:
			category_to_filenames[category] = []

		filename,_ = os.path.splitext(row['filename'])
		category_to_filenames[category].append(filename)
	# print category_to_filenames

	overlap_threshold = 0.5
	for category in category_to_filenames:
		print category
		results = []
		for scored_data_name in scored_datas:
			# print scored_data_name
			average_precision = 0.0
			average_recall = 0.0
			scored_data = scored_datas[scored_data_name]
			filtered_filenames = [k for k in scored_data.keys() if k in category_to_filenames[category]]
			for file_name in filtered_filenames:
				predicted_list = scored_data[file_name]
				ground_truth = ground_truth_data[file_name]
				matches = np.zeros((len(ground_truth)))
				false_negative = 0
				false_postive = 0
				true_positive = 0
				for p in predicted_list:
					best_overlap = -1000
					best_match = None
					for g_i, g in enumerate(ground_truth):
						current_overlap  = bb_intersection_over_union(p, g)
						if current_overlap > best_overlap:
							best_overlap = current_overlap
							best_match = g_i
					if not best_match is None and best_overlap >= overlap_threshold:
						matches[best_match] += 1
					else:
						false_postive += 1

				for g in matches:
					if g > 0:
						true_positive += 1
						false_postive += int(g)-1
					else:
						false_negative += 1

				precision = 0.0
				recall = 0.0
				if (true_positive > 0 or false_postive > 0):
					precision = float(true_positive) / (true_positive + false_postive)
				if (true_positive > 0 or false_negative > 0):
					recall = float(true_positive) / (true_positive + false_negative)
				average_precision += precision
				average_recall += recall
				# print 'TP: [%d] FP:[%d] FN: [%d]' % (true_positive, false_postive, false_negative)
				# print 'P: [%f] R: [%f]' % (precision, recall)
				# print predicted_list
				# print ground_truth
			average_precision /= len(filtered_filenames)
			average_recall /= len(filtered_filenames)
			f1 = 0.0
			if (average_precision + average_recall) > 0:
				f1 = 2 * (average_precision * average_recall) / (average_precision + average_recall)
			results.append(f1)
		df[category] = results
	# 		# print 'AP: [%f] AR: [%f]' % (average_precision, average_recall)

	print df
	df.to_csv(os.path.join(dir_path,'..','results-by-category.csv'))
	# print scored_data


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou


if __name__ == "__main__":
	score_detections()