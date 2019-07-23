python3 convert_xml_to_csv.py

echo "Generate TFRecord for train dataset"
python generate_tfrecord.py --label0=square --label1=triangle --csv_input=../data/train_labels.csv --img_path=../data/train  --output_path=../data/train.record

echo "Generate TFRecord for test dataset"
python generate_tfrecord.py --label0=square --label1=triangle --csv_input=../data/test_labels.csv --img_path=../data/test  --output_path=../data/test.record

echo "Your datasets are ready for transfer learning"
echo "-- DONE --"