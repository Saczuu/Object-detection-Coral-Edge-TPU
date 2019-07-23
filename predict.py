import argparse
import platform
import subprocess
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

	parser = argparse.ArgumentParser(description="Make prediction object 		detection on input images")
	requiredArgs = parser.add_argument_group("required argument")
	requiredArgs.add_argument("--input", dest="input", type=str,
                   help="path to image for prediciton",
                   required = True)
	requiredArgs.add_argument("--show", dest="show", type=bool,
		   help="If show prediciton effect",
		   required = True)
	args = parser.parse_args()
	input = args.input

  	# Initialize engine.
  	engine = DetectionEngine("model/output/output_compiled.tflite")
  	labels = ReadLabelFile("model/output/labels.txt)

  	# Open image.
  	img = Image.open(input)
  	draw = ImageDraw.Draw(img)

  	# Run inference.	
  	ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,
                               relative_coord=False, top_k=10)

	# Display result.
	img = cv2.imread(input)
	colors = {"square":(0,255,0), "triangle":(255,0,0)}
  	if ans:
  		for obj in ans:
      			print ('-----------------------------------------')
      			if labels:
        			print(labels[obj.label_id])
      			print ('score = ', obj.score)
      			box = obj.bounding_box.flatten().tolist()
      			print ('box = ', box)
			if arg.show == True:
				cv2.rectangle(img, box[0], box[3], 					      color[labels[obj.label_id]], 3)
	if arg.show = True:
		cv2.imshow('prediction', img)


