from Model.model import AutoEncoder
import os
import cv2
import numpy as np
import time

size = (200, 200)
batch_sizes = [4]

inputs_folder = 'Data/Motion_Blur_Noise'
targets_folder = 'Data/Distorted'

inputs = []
targets = []
names = []
for n, image_path in enumerate(os.listdir(inputs_folder)):
  if n<6000:
    image = cv2.imread(os.path.join(inputs_folder, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, size)
    input = np.reshape(image, [size[0], size[1], 1]) / 255
    image = cv2.imread(os.path.join(targets_folder, image_path))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, size)
    target = np.reshape(image, [size[0], size[1], 1]) / 255
    inputs.append(input)
    names.append(image_path)
    targets.append(target)

test_imgs = inputs[-600:]
inputs = inputs[:-600]
test_trgs = targets[-600:]
targets = targets[:-600]

PROGRESS_BAR_LEN = 25
def print_progress(i, total, elapsed_time):
  progress = (i+1)/float(total)
  bar = 'Progress' + ': |'
  bar += ''.join(['█' for _ in range(int(PROGRESS_BAR_LEN * min(1, progress)))])
  bar += ''.join([' ' for _ in range(int(PROGRESS_BAR_LEN * max(0, (1 - progress))))])
  bar += '| {percentage}%'.format(percentage=int(progress * 100))
  bar += ' - {completed}/{total}'.format(completed=i+1, total=total)
  bar += ' - {min} min {sec} sec'.format(min=int(elapsed_time / 60.),
                                         sec=int(elapsed_time - int(elapsed_time / 60.) * 60))
  print("\033[F" + bar + '')

#before
filters = [[32, 16], [8, 4], [64], [32], [16], [8]]
filters = [[32, 16], [16, 8]] #, [64], [32], [16], [8]]

for batch_size in batch_sizes:
  for fn, filter in enumerate(filters):
    model_path = "./CNN_"+str(fn)+"_Batch_"+str(batch_size)
    configs = {
      "input_size": size,
      "filters_number": filter,
      "filters_size": [(5, 5) for _ in range(len(filter))]
    }
    model = AutoEncoder(configs)

    try:
     for epoch in range(200):
      begin = time.time()
      print("Beginning epoch n. {}\n".format(epoch+1))
      for index in range(0, len(inputs), batch_size):
        model.train(inputs[index:index+batch_size], targets[index:index+batch_size])
        print_progress(index/batch_size+1, len(inputs)/batch_size+1, time.time()-begin)
      print("Epoch finished. Time utilized for this epoch: {}".format(int(time.time()-begin)))
      if (epoch+1)%50 == 0:
        model.save(model_path+'_'+str(epoch+1))
        for i, img in enumerate(test_imgs):
          output = np.array(model.infer([img])[0])[0]*255
          output_folder = os.path.join(model_path+'_'+str(epoch+1), 'Results')
          if not os.path.exists(output_folder):
            os.makedirs(output_folder)
          output = np.concatenate((img*255, output), axis=1)
          output = np.concatenate((output, test_trgs[i]*255), axis=1)
          cv2.imwrite(os.path.join(output_folder, names[i]), output)

    except KeyboardInterrupt:
      print("Exiting...")

    print("Training process is over, have a look on the results")
    model.save(model_path)

    del model

