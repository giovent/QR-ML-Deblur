import numpy as np
import cv2

from Model.model import AutoEncoder

import time

from pyzbar.pyzbar import decode

# Ititialize model
configs = {'model_path': 'Model/Saved Model',
           'input_size': (200, 200)}
model = AutoEncoder(configs=configs,from_zero=False)
cam = cv2.VideoCapture(1)


both_no = 0
both_yes = 0
mine_yes = 0
orig_yes = 0

for i in range(1000):
  ret, img = cam.read()
  if i==0:
    h, w = img.shape[0], img.shape[1]
    print(h, w)
  img = img[:, int((w-h)/2):int(h+(w-h)/2)]
  #img = img[160:360, 500:700]

  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.resize(img, configs['input_size'])
  img = np.reshape(img, [configs['input_size'][0], configs['input_size'][1], 1])/255

  a = time.time()
  result = np.array(model.infer([img])[0])[0]
  cv2.imshow('Original', img)
  cv2.imshow('Result', result)

  d1 = decode(result*255)
  d2 = decode(img*255)

  if len(d1) == 0:
    d1 = 0
  else:
    d1 = int(d1[0].data == b'http://fpj.datarj.com/einv/fm?q=b249MTIwMjIwMTkyMDAxNDAzMjIzMTA1JnNpPWQwNzVhMWE3MjI5NTRlMWQ2NWNjOWUwNjlkODNjM2Qw\r\n                    ')

  if len(d2) == 0:
    d2 = 0
  else:
    d2 = int(d2[0].data == b'http://fpj.datarj.com/einv/fm?q=b249MTIwMjIwMTkyMDAxNDAzMjIzMTA1JnNpPWQwNzVhMWE3MjI5NTRlMWQ2NWNjOWUwNjlkODNjM2Qw\r\n                    ')


  if d1==0 and d2==0:
    both_no += 1
  elif d1>0 and d2>0:
    both_yes += 1
  elif d1>0 and d2==0:
    mine_yes += 1
  elif d1==0 and d2>0:
    orig_yes += 1


  print("Win: {}, Lost {}, Same {}".format(mine_yes, orig_yes, both_yes+both_no))
  cv2.waitKey(1)


