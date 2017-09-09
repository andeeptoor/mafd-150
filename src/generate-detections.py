import os 
import pandas as pd
import json
from detectors import *

def generate_detections():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	image_dir = os.path.join(dir_path,"..","images")

	print 'Reading in dataset...'
	dataset_file = os.path.join(dir_path,"..","dataset.csv")
	dataset_df = pd.read_csv(dataset_file)

	print "Load detectors..."
	detectors = {'dlib':DlibFaceDetector(image_dir), 'google':GoogleFaceDetector(), 'opencv':OpenCVFaceDetector(image_dir),'microsoft':MicrosoftFaceDetector()}
	for _, row in dataset_df.iterrows():
		url = row['url']
		file = row['filename']
		for detector_name in detectors:
			print "Running [%s]..." % (detector_name)
			results_dir = os.path.join(dir_path,"..","results/" + detector_name)
			results_file = os.path.join(results_dir, file + ".json")
			if os.path.exists(results_file):
				continue
			else:
				print '\tReading [%s] from [%s]...' % (file, url)
			detector = detectors[detector_name]
			results = detector.detect_faces(row)
			with open(results_file, "w") as output_file:
				json.dump(results, output_file)

if __name__ == "__main__":
	generate_detections()
	