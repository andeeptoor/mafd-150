import time

import dlib
from skimage import io

class DlibFaceDetector:

	def __init__(self, image_dir):
		self.detector = dlib.get_frontal_face_detector()
		self.image_dir = image_dir

	def detect_faces(self,image_data):
		uri = image_data['url']
		filename = image_data['filename']
		results = {'url':uri,'faces':[]}
		file_path = os.path.join(self.image_dir, filename)
		# print file_path
		img = io.imread(file_path)
		dets = self.detector(img, 1)
		for d in dets:
			face_results = {'xmin':d.left(), 'ymin':d.top(), 'xmax':d.right(), 'ymax':d.bottom()}
			results['faces'].append(face_results)
		return results
	

import os
import cv2

class OpenCVFaceDetector:
	def __init__(self, image_dir):
		opencv_home = '/sb/built/opencv-3.2.0/data'
		front_detector = cv2.CascadeClassifier(os.path.join(opencv_home, 'haarcascades/haarcascade_frontalface_default.xml'))
		profile_detector = cv2.CascadeClassifier(os.path.join(opencv_home, 'haarcascades/haarcascade_profileface.xml'))
		self.detectors = {'front':front_detector, 'profile':profile_detector}
		self.image_dir = image_dir
	
	def detect_faces(self,image_data):
		uri = image_data['url']
		filename = image_data['filename']
		results = {'url':uri,'faces':[]}
		try:
			img = cv2.imread(os.path.join(self.image_dir, filename))
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		except:
			return results

		for face_detector_name in self.detectors:
			face_detector = self.detectors[face_detector_name]
			faces = face_detector.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
				face_results = {'xmin':int(x), 'ymin':int(y), 'xmax':int(x+w), 'ymax':int(y+h)}
				face_results['name'] = face_detector_name
				results['faces'].append(face_results)
		return results

from google.cloud import vision
from google.cloud.vision import types

class GoogleFaceDetector:

	def detect_faces(self,image_data):
	    """Detects faces in the file located in Google Cloud Storage or the web."""
	    uri = image_data['url']
	    client = vision.ImageAnnotatorClient()
	    image = types.Image()
	    image.source.image_uri = uri

	    response = client.face_detection(image=image)
	    faces = response.face_annotations

	    # print('Faces:')

	    results = {'url':uri,'faces':[]}
	    for face in faces:
	    	# print face
	        vertices = (['({},{})'.format(vertex.x, vertex.y)
	                    for vertex in face.bounding_poly.vertices])

	        face_results = {'xmin':face.bounding_poly.vertices[0].x, 'ymin':face.bounding_poly.vertices[0].y, 'xmax':face.bounding_poly.vertices[2].x, 'ymax':face.bounding_poly.vertices[2].y}
	        face_results['confidence'] = face.detection_confidence
	        results['faces'].append(face_results)
	    time.sleep(5)
	    return results

import httplib, urllib, base64, json

class MicrosoftFaceDetector:
	def __init__(self):
		key = 'ad978e94e911490389052ee9db01b4dd'  # Replace with a valid subscription key (keeping the quotes in place).
		uri_base = 'westcentralus.api.cognitive.microsoft.com'
		self.headers = {
		    'Content-Type': 'application/json',
		    'Ocp-Apim-Subscription-Key': key,
		}
		self.params = urllib.urlencode({
		    'returnFaceId': 'false',
		    'returnFaceLandmarks': 'false',
		    'returnFaceAttributes': '',
		})

		
	def detect_faces(self,image_data):
		uri = image_data['url']
		results = {'url':uri,'faces':[]}
		
		body = "{'url':'%s'}" % (uri)

		try:
		    # Execute the REST API call and get the response.
		    conn = httplib.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
		    conn.request("POST", "/face/v1.0/detect?%s" % self.params, body, self.headers)
		    response = conn.getresponse()
		    data = response.read()

		    # 'data' contains the JSON data. The following formats the JSON data for display.
		    result = json.loads(data)
		    conn.close()

		except Exception as e:
			print e
			return results

		for face in result:
			x = face['faceRectangle']['left']
			y = face['faceRectangle']['top']
			face_results = {'xmin':int(x), 'ymin':int(y), 'xmax':int(x+face['faceRectangle']['width']), 'ymax':int(y+face['faceRectangle']['height'])}
			results['faces'].append(face_results)
		time.sleep(5)
		return results