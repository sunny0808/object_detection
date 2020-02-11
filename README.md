# Object detetion in self drive
kitti2xml.py:
(1).Resize kitti image to 1216x352 jpg format
(2).Change kitti annotation txt file to PASCAL VOC xml file
(3).Draw box in resized image according to xml file
Be sure to set directory in first few lines

generate_train_validation.py:
Generate train and validation data set(file name) in PASCAL
VOC format.
