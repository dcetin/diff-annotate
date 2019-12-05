import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from sklearn.cluster import KMeans
from skimage.measure import compare_ssim
from imageai.Detection import ObjectDetection

# # Snippets taken from stackoverflow are as follows:
# resizeWithAspectRatio: https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
# brightness:            https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
# propCreator:           https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument("--day", help="The name of the directory to inspect (e.g. date in YYYY-MM-DD format for example dataset).", required=True)
parser.add_argument("--startIdx", help="Starting index for iteration.", type=int, default=0)
parser.add_argument("--display", help="Image to be displayed to user.", choices=["collage", "proposal", "comparison"], default="comparison")
parser.add_argument("--excludeDark", help="Run only on the bright images separated from the dark ones.", action="store_true")
parser.add_argument("--saveCSV", help="Save dataframe as csv upon exiting.", action="store_true")
# ROI extraction and bounding boxes
parser.add_argument("--differenceThresh", help="Pixel intensity value to threshold image differences.", type=int, default=100)
parser.add_argument("--redundancyThresh", help="Threshold for eliminating redundant ROIs with neighboring centers.", type=int, default=50)
parser.add_argument("--areaMin", help="Minimum allowed area for the ROIs.", type=int, default=3)
parser.add_argument("--areaMax", help="Maximum allowed area for the ROIs.", type=int, default=80)
parser.add_argument("--radius", help="Padding width while drawing the bounding boxes.", type=int, default=7)
# Detector
parser.add_argument("--useDetector", help="Run object detectors to identify people.", action="store_true")
parser.add_argument("--detModel", help="Detection model to use.", choices=["retinanet", "yolo", "yolo-tiny"], default="retinanet")
parser.add_argument("--detThresh", help="Detection confidence threshold.", type=int, default=15)
args = parser.parse_args()

def brightness(im_file):
  '''Returns boolean arrays for bright and dark clusters.'''
  im = Image.open(im_file).convert("L")
  stat = ImageStat.Stat(im)
  return stat.mean[0]

def groupDayNight(avgLights):
  '''Computes the average brightness.'''
  kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(avgLights).reshape(-1, 1))
  grp1 = (kmeans.labels_ == 1)
  grp0 = (kmeans.labels_ == 0)
  if(avgLights[grp1].mean() > avgLights[grp0].mean()):
      return grp1, grp0
  return grp0, grp1

def propCreator(other, middle, proposal, rois, threshVal, name):
  '''Create proposals given pairs of images.'''
  middle_gray = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)
  other_gray  = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)

  # Compute SSIM between two images
  (score, diff) = compare_ssim(other_gray, middle_gray, full=True, multichannel=True)
  diff = (diff * 255).astype("uint8")
  cv2.imwrite(os.path.join(pathTemp, "diff-" + name + ".jpg"), diff)
  thresh = cv2.threshold(diff, threshVal, 255, cv2.THRESH_BINARY_INV)[1]
  cv2.imwrite(os.path.join(pathTemp, "thresh-" + name + ".jpg"), thresh)
  contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  for c in contours:
      area = cv2.contourArea(c)
      if area > args.areaMin and area < args.areaMax:
          x,y,w,h = cv2.boundingRect(c)
          x1, y1, x2, y2 = x-args.radius, y-args.radius, x+w+args.radius, y+h+args.radius
          # cv2.rectangle(proposal, (x1, y1), (x2, y2), (0,255,255), 1)
          roiEdges = (x1, x2, y1, y2)
          excludeROI = False
          # rois.append(middle[y1:y2, x1:x2])
          for i in rois:
              diffSum = sum([abs(a-b) for (a,b) in zip(i, roiEdges)])
              if diffSum < args.redundancyThresh:
                  excludeROI = True
                  break
          if not excludeROI:
              rois.append(roiEdges)
  return rois, proposal

def getProp(i, detector=None):
  '''Extract ROIs for the given image index.'''
  middlePath = imgNames[i]
  beforePath = imgNames[i-1] if i>0 else middlePath
  afterPath  = imgNames[i+1] if i<len(imgNames) else middlePath
  middlePath = os.path.join(pathDay, middlePath)
  beforePath = os.path.join(pathDay, beforePath)
  afterPath = os.path.join(pathDay, afterPath)
  before = cv2.imread(beforePath)
  middle = cv2.imread(middlePath)
  after  = cv2.imread(afterPath)
  proposal = middle.copy()

  rois = []
  rois, proposal = propCreator(before, middle, proposal, rois, args.differenceThresh, name="before")
  rois, proposal = propCreator(after,  middle, proposal, rois, args.differenceThresh, name="after")
  [cv2.rectangle(proposal, (x1, y1), (x2, y2), YELLOW, 1) for (x1, x2, y1, y2) in rois]

  colorImage  = Image.open(middlePath)
  rotated     = colorImage.rotate(90, expand=True)
  rotated.save(os.path.join(pathTemp, "temp.jpg"))

  if detector:
      detections = detector.detectObjectsFromImage(
          input_image=os.path.join(pathTemp, "temp.jpg"),
          output_image_path=os.path.join(pathTemp , "output.jpg"),
          minimum_percentage_probability=args.detThresh)
      for eachObject in detections:
          if eachObject["name"] == "person":
              print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
              (x1, y1, x2, y2) = tuple(eachObject["box_points"])
              x1, x2, y1, y2 = x1-args.radius, x2+args.radius, y1-args.radius, y2+args.radius
              cv2.rectangle(proposal, (x1, y1), (x2, y2), RED, 1)

  if rois:
      rois = [middle[y1:y2, x1:x2] for (x1, x2, y1, y2) in rois]
  tbefore = cv2.imread(os.path.join(pathTemp, "thresh-before.jpg"))
  tafter = cv2.imread(os.path.join(pathTemp, "thresh-after.jpg"))
  sep = np.zeros((middle.shape[0], 5, 3))
  top = np.hstack((before, sep, middle, sep, after))
  bot = np.hstack((tbefore, sep, proposal, sep, tafter))
  collage = np.vstack((top, bot))
  comparison = np.hstack((before, sep, proposal, sep, after))
  cv2.imwrite(os.path.join(pathTemp, "proposal.jpg"), proposal)
  cv2.imwrite(os.path.join(pathTemp, "collage.jpg"), collage)
  cv2.imwrite(os.path.join(pathTemp, "comparison.jpg"), comparison)
  return imgNames[i], proposal

def resizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
  '''Resize named cv2 window while keeping the aspect ratio.'''
  dim = None
  (h, w) = image.shape[:2]
  if width is None and height is None:
    return image
  if width is None:
    r = height / float(h)
    dim = (int(w * r), height)
  else:
    r = width / float(w)
    dim = (width, int(h * r))
  return cv2.resize(image, dim, interpolation=inter)

def getDetModel(modelName):
  '''Get detector model given name.'''
  detector = ObjectDetection()
  if modelName == "retinanet":
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(pathModel , "resnet50_coco_best_v2.0.1.h5"))
  elif modelName == "yolo":
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(pathModel , "yolo.h5"))
  elif modelName == "yolo-tiny":
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath( os.path.join(pathModel , "yolo-tiny.h5"))
  detector.loadModel()
  return detector

def getImages(pathDay, excludeDark=False, startIdx=0):
  '''Get sorted list of images in the specified directory.'''
  imgNames = sorted([f for f in os.listdir(pathDay) if os.path.isfile(os.path.join(pathDay, f))])
  dayIdx = []
  if args.excludeDark:
    imgPaths = [os.path.join(pathDay, img) for img in imgNames]
    avgLights = [brightness(img) for img in imgPaths]
    avgLights = np.array(avgLights)
    grpDay, grpNight = groupDayNight(avgLights)
    dayImages = np.array(imgNames)[grpDay].tolist()
    nightImages = np.array(imgNames)[grpNight].tolist()
    print("Labeled day and night images.")
    dayIdx = np.where(grpDay == True)[0].tolist()

  imgList = range(args.startIdx, len(imgNames))
  if args.excludeDark:
    imgList = dayIdx[args.startIdx:]

  return imgList, imgNames

# Set paths
pathData = os.path.join(os.getcwd(), "data")
pathModel = os.path.join(os.getcwd(), "models")
pathTemp = os.path.join(os.getcwd(), "temp")
pathDay = os.path.join(pathData, args.day)
if not os.path.exists(pathTemp):
    os.makedirs(pathTemp)

# BGR constants for bounding boxes
RED = (0,0,255)
YELLOW = (0,255,255)

# Assign labels to keys, [QUIT] ceases iteration
labelDict = json.load(open(os.path.join(os.getcwd(), "labelDict.json")))

# Get detector model
detector = getDetModel(args.detModel) if args.useDetector else None

# Get images
imgList, imgNames = getImages(pathDay, excludeDark=args.excludeDark, startIdx=args.startIdx)

doneImages = []
doneLabels = []

for i in imgList:
  imgname, proposal = getProp(i, detector)
  winname = "proposal"
  keyInDict = False
  while(1):
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, 0,0)
    img = cv2.imread(os.path.join(pathTemp, args.display + ".jpg"))
    img = resizeWithAspectRatio(img, height=650, width=1200)
    cv2.imshow(winname, img)

    keyPressed = cv2.waitKey(0)
    for key, label in labelDict.items():
      if keyPressed == ord(key):
        if label != "[QUIT]":
          doneImages.append(imgname)
          doneLabels.append(label)
          print("Image {} [{}] labeled as {}".format(i, imgNames[i], label))
        keyInDict = True
        break
    if keyInDict:
      break
  if label == "[QUIT]":
    break

df = pd.DataFrame(list(zip(doneImages, doneLabels)), columns =["Image", "Label"])

if args.saveCSV:
  df.to_csv("{}_{}.csv".format(args.day, args.startIdx), index=False)
