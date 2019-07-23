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
	print("-- Staring --")
	parser = argparse.ArgumentParser(description="Make prediction object 		detection on input images")
	requiredArgs = parser.add_argument_group("required argument")
	requiredArgs.add_argument("--input", dest="input", type=str,help="path to image for prediciton",required = True)
	requiredArgs.add_argument("--show", dest="show", type=bool,help="If show prediciton effect",required = True)
	args = parser.parse_args()
	input = args.input

  	# Initialize engine.
	engine = DetectionEngine("model/output/output_compiled.tflite")
	labels = ReadLabelFile("model/output/labels.txt")

  	# Open image.
	img = Image.open(input)
	print(type(img))
	draw = ImageDraw.Draw(img)

  	# Run inference.
	print("-- Forward propagation --")	
	ans = engine.DetectWithImage(img, threshold=0.05, keep_aspect_ratio=True,       relative_coord=False, top_k=10)

	# Display result.
	img = cv2.imread(input)
	colors = {"Square":(0,255,0), "Triangle":(255,0,0)}
	if ans:
		for obj in ans:
			print ('-----------------------------------------')
			if labels:
				print(labels[obj.label_id])
			print ('score = ', obj.score)
			box = obj.bounding_box.flatten().tolist()
			print ('box = ', box)
			if args.show == True:
				cv2.rectangle(img,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),colors[labels[obj.label_id]],3)
				cv2.putText(img, labels[obj.label_id], (int(box[0])+5,int(box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255,255,255), 1, cv2.LINE_AA)
	if args.show == True:
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


if __name__ == '__main__':
  main()
