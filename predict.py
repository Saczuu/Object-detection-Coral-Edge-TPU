import argparse

parser = argparse.ArgumentParser(description="Make prediction object detection on input images")
requiredArgs = parser.add_argument_group("required argument")
requiredArgs.add_argument("--input", dest="input", type=str,
                   help="path to image for prediciton",
                   required = True)
args = parser.parse_args()
input = args.input

print(input)