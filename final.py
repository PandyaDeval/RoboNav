from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import urllib.request
import cv2
import numpy as np
import time
from sklearn.preprocessing import normalize

url_l='http://192.168.43.88:8080/photo.jpg'
url_r='http://192.168.43.1:8080/photo.jpg'

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def stitch_images(images):
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    return (status,stitched)


class camera:
    def __init__(self, url, identity):
        self.url = url
        self.identity = identity


    def get_current_frame(self):
        imgResp = urllib.request.urlopen(self.url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        return img


def crop_stitched(stitched):
    # check to see if we supposed to crop out the largest rectangular
    # region from the stitched image
    # create a 10 pixel border surrounding the stitched image
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
                                    cv2.BORDER_CONSTANT, (0, 0, 0))

    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # create two copies of the mask: one to serve as our actual
    # minimum rectangular region and another to serve as a counter
    # for how many pixels need to be removed to form the minimum
    # rectangular region

    minRect = mask.copy()
    sub = mask.copy()

    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)
    stitched = stitched[y:y + h, x:x + w]
    return stitched


def calc_depth(images):
    imgL = images[0]
    imgR = images[1]

    window_size = 3
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=160,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
    #cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    return filteredImg

images_l = []
images_r = []

camera_left = camera(url_l, 'left')
camera_right = camera(url_r, 'right')
# Set FPS using t

t = 0.05
i = 0
br=0
while True:
    inp = input()
    if inp == 'k':

        # Use urllib to get the image and convert into a cv2 usable format
        t1 = time.time()

        img_l = camera_left.get_current_frame()
        img_r = camera_right.get_current_frame()
        img_l[img_l == 0] = 20
        img_r[img_r == 0] = 20
        # put the image on screen
        cv2.imshow('IPWebcam_l', img_l)
        cv2.imshow('IPWebcam_r', img_r)

        # To give the processor some less stress
        # time.sleep(0.1)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        images_l.append(img_l)
        images_r.append(img_r)

        print(i)
        i += 1
        t2 = time.time()
        if (t2 - t1 < t):
            time.sleep(t - t2 + t1)

    elif inp == 'q':
        break

print("[INFO] stitching images...")

(status_r, stitched_r) = stitch_images(images_r)
(status_l, stitched_l) = stitch_images(images_l)

# if the status is '0', then OpenCV successfully performed image
# stitching
if status_r == 0 and status_l == 0:
    # write the output stitched image to disk
    # cv2.imwrite(args["output"], stitched)

    # display the output stitched image to our screen
    '''print(np.shape(stitched_l)[1], np.shape(stitched_r)[1])
    w=min(np.shape(stitched_l)[1],np.shape(stitched_r)[1])
    print(w)
    print(np.shape(stitched_l)[0], np.shape(stitched_r)[0])
    h=min(np.shape(stitched_l)[0],np.shape(stitched_r)[0])
    print(h)
    stitched_l = image_resize(stitched_l,width=w,height=h)
    stitched_r = image_resize(stitched_r,width=w,height=h)
    cv2.imshow("Stitched_l", stitched_l)
    cv2.imshow("Stitched_r", stitched_r)
    print(np.shape(stitched_l)[1], np.shape(stitched_r)[1])
    print(np.shape(stitched_l)[0], np.shape(stitched_r)[0])
    cv2.waitKey(0)'''


    print("[INFO] cropping...")
    #stitched_r = crop_stitched(stitched_r)
    #stitched_l = crop_stitched(stitched_l)

    # display the output stitched image to our screen
    cv2.imshow("Stitched_l", stitched_l)
    cv2.imshow("Stitched_r", stitched_r)

    #cv2.imwrite("pano_r.jpg", stitched_r)
    cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints)
# being detected
else:
    print("[INFO] image stitching failed (right = {}, left = {})".format(status_l,status_r))
try:
    min_h = min(stitched_r.shape[0], stitched_l.shape[0])
    min_w = min(stitched_r.shape[1], stitched_l.shape[1])
except:
    print("Couldn't stitch,retry!")
print(stitched_r.shape,stitched_l.shape)
print(min_h,min_w)
stitched_l=cv2.resize(stitched_l, (min_w,min_h))
stitched_r=cv2.resize(stitched_r, (min_w,min_h))
print(stitched_r.shape,stitched_l.shape)
cv2.imwrite('input_l.jpg',stitched_l)
cv2.imwrite('input_r.jpg',stitched_r)

filteredImg=calc_depth([stitched_l,stitched_r])
filteredImg[filteredImg!=[0]]+=50
filteredImg= filteredImg%255
print('loading images...')
cv2.namedWindow('Disparity Map', cv2.WINDOW_NORMAL)
cv2.imshow('Disparity Map', filteredImg)
cv2.waitKey(0)
'''
from imageai.Detection import ObjectDetection
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=stitched_l , output_image_path=os.path.join(execution_path , "imagenew.jpg"))
cv2.imshow('Image', imagenew.jpg)
cv2.waitKey(0)

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

detections, extracted_images = detector.detectObjectsFromImage(stitched_l, output_image_path=os.path.join(execution_path , "inputnew.jpg"), extract_detected_objects=True)'''