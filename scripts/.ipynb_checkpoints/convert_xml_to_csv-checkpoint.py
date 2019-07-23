import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET

def xml_to_csv(path_to_xml, path_for_output):
    xml_list = []
    for xml_file in glob.glob(path_to_xml + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return  xml_df.to_csv(
        path_for_output, index=None)

#CSV for train images
print("-- Converting train data --")
xml_to_csv("../data/train", "../data/train_labels.csv")

#CSV for test images
print("-- Converting test data --")
xml_to_csv("../data/test", "../data/test_labels.csv")

print("-- CONVERSION INTO CSV DONE --")