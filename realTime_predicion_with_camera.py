import argparse
import platform
import subprocess
import time
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw
import cv2




# Function to read labels from text files.
def ReadLabelFile(file_path):
	with open(file_path, 'r', encoding="utf-8") as f:
		lines = f.readlines()
	ret = {}
	for line in lines:
		pair = line.strip().split(maxsplit=1)
		ret[int(pair[0])] = pair[1].strip()
	return ret


def main():
	print("-- Staring --")

  	# Initialize engine.
	engine = DetectionEngine("model/output/output_compiled.tflite")
	labels = ReadLabelFile("model/output/labels.txt")

	print("-- Start recording --")
	# Initialize camera
	cap = cv2.VideoCapture(0)
	colors = {"Square":(0,255,0), "Triangle":(255,0,0)}

	while(True):
		ret, frame = cap.read(0)

  		# Run inference.
		img = Image.fromarray(frame)	
		ans = engine.DetectWithImage(img,keep_aspect_ratio=True,relative_coord=False)

		# Display result.
		if ans:
			for obj in ans:
				box = obj.bounding_box.flatten().tolist()
				cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colors[labels[obj.label_id]],3)
				cv2.putText(frame, labels[obj.label_id], (int(box[0])+5,int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255,255,255), 1, cv2.LINE_AA)
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
    
	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
