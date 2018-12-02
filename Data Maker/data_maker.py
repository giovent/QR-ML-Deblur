import qrcode
import os
import numpy as np
import cv2

class QR_Maker:
  def __init__(self):
    self.qr = qrcode.QRCode(1)

  def set_data(self, string):
    self.data = string
    self.qr.clear()
    self.qr.add_data(self.data)

  def save_on_file(self, directory='', file_path=None):
    self.qr.make(fit=True)
    img = self.qr.make_image(fill_color="black", back_color="white")

    if file_path is None:
      file_path = str(self.data) + '.png'

    if not os.path.exists(directory):
      os.makedirs(directory)

    img.save(os.path.join(directory, file_path))

  def make_QRs(self, how_many, data_length, out_folder):
    for i in range(how_many):
      data = ''.join([str(np.random.randint(0, 9)) for _ in range(data_length)])
      self.set_data(data)
      self.save_on_file(out_folder, str(data) + '.png')

def rnd(low=0, high=30):
  return np.random.randint(low, high, 1)


class DataProcessor:
  def __init__(self):
    pass

  def distort(self, img, out_size):
    H, W = out_size
    h, w = img.shape[0], img.shape[1]
    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = np.float32([[rnd(), rnd()], [rnd(), W-rnd()], [H-rnd(), rnd()], [H-rnd(), W-rnd()]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (H, W))

  def distort_from_folder(self, in_folder, out_folder, out_size):
    if not os.path.exists(in_folder):
      return
    for img_path in os.listdir(in_folder):
      img = cv2.imread(os.path.join(in_folder, img_path))
      dst = self.distort(img, out_size)
      if not os.path.exists(out_folder):
        os.makedirs(out_folder)
      cv2.imwrite(os.path.join(out_folder, img_path), dst)

  def median_blur(self, img, level):
    return cv2.medianBlur(img, level)

  def gaussian_blur(self, img):
    return cv2.GaussianBlur(img, (5, 5), 5)

  def blur_from_folder(self, in_folder, out_folder):
    if not os.path.exists(in_folder):
      return
    for img_path in os.listdir(in_folder):
      img = cv2.imread(os.path.join(in_folder, img_path))
      #dst = self.median_blur(img, 5)
      dst = self.gaussian_blur(img)
      if not os.path.exists(out_folder):
        os.makedirs(out_folder)
      cv2.imwrite(os.path.join(out_folder, img_path), dst)

  def add_noise(self, img, level=15):
    h, w = img.shape[0], img.shape[1]
    img = img + np.random.randn(h, w, 3) * level
    return img

  def add_noise_from_folder(self, in_folder, out_folder):
    if not os.path.exists(in_folder):
      return
    for img_path in os.listdir(in_folder):
      img = cv2.imread(os.path.join(in_folder, img_path))
      dst = self.add_noise(img)
      if not os.path.exists(out_folder):
        os.makedirs(out_folder)
      cv2.imwrite(os.path.join(out_folder, img_path), dst)

  def motion_blur(self, img, size=10):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)*np.random.rand()
    kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)*np.random.rand()
    kernel_motion_blur = kernel_motion_blur / size
    return cv2.filter2D(img, -1, kernel_motion_blur)

  def motion_blur_from_folder(self, in_folder, out_folder):
    if not os.path.exists(in_folder):
      return
    for img_path in os.listdir(in_folder):
      img = cv2.imread(os.path.join(in_folder, img_path))
      dst = self.motion_blur(img)
      if not os.path.exists(out_folder):
        os.makedirs(out_folder)
      cv2.imwrite(os.path.join(out_folder, img_path), dst)



if __name__=='__main__':
  for length in range(50, 151, 5):
    QR_Maker().make_QRs(how_many=500, data_length=length, out_folder='../Data/Original')

  DataProcessor().distort_from_folder('../Data/Original', '../Data/Distorted', [220, 220])
  DataProcessor().add_noise_from_folder('../Data/Distorted', '../Data/Noisy')
  #DataProcessor().blur_from_folder('../Data/Distorted', '../Data/Blurred')
  DataProcessor().motion_blur_from_folder('../Data/Noisy', '../Data/Motion_Blur')