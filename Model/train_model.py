from Model.model import AutoEncoder
import os
import cv2
import numpy as np

configs = {
  "input_size": (200, 200),
  "filters_number": [32, 32, 32, 16],
  "filters_size": [(5, 5) for _ in range(4)]
}

model = AutoEncoder(configs)

inputs_folder = 'Data/Motion_Blur'
targets_folder = 'Data/Original'

inputs = []
targets = []
for image_path in os.listdir(inputs_folder):
  image = cv2.imread(os.path.join(inputs_folder, image_path))
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image = cv2.resize(image, (200, 200))
  input = np.reshape(image, [200,200,1])/255
  image = cv2.imread(os.path.join(targets_folder, image_path))
  image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  image = cv2.resize(image, (200, 200))
  target = np.reshape(image, [200,200,1])/255
  inputs.append(input)
  targets.append(target)

batch_size = 100
print len(inputs)

for epoch in range(10):
  print("Beginning epoch n. {}".format(epoch+1))
  for index in range(0, len(inputs), batch_size):
    model.train(inputs[index:index+batch_size], targets[index:index+batch_size])

print("Training process is over")
cv2.imshow('Input', inputs[0])
cv2.imshow('Target', targets[0])
cv2.imshow('Result', np.array(model.infer([inputs[0]])[0][0]))
cv2.waitKey()