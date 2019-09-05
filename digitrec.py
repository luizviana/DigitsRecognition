#By Luiz Viana
import cv2
import numpy as np

# FUNCTIONS
def nothing(x):
    pass


digits = cv2.imread('digits.png', 0)

rows = np.vsplit(digits, 50)
cells = []
for row in rows:
    row_cell = np.hsplit(row, 50)
    for cell in row_cell:
        cell = cell.flatten()
        cells.append(cell)

cells = np.array(cells, dtype=np.float32)

k = np.arange(10)
cells_labels = np.repeat(k, 250)

knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)

# CREATE TRACKBARS WINDOW
cv2.namedWindow('track')

# CREATE TRACKBARS
cv2.createTrackbar('x', 'track', 140, 800, nothing)
cv2.createTrackbar('y', 'track', 105, 800, nothing)
cv2.createTrackbar('w', 'track', 420, 800, nothing)
cv2.createTrackbar('h', 'track', 330, 800, nothing)
cv2.createTrackbar('th', 'track', 51, 255, nothing)

# CAPTURE WEBCAM
cap = cv2.VideoCapture(0)

# LOOP
while(True):

    # GET TRACKBAR
    x = cv2.getTrackbarPos('x', 'track')
    y = cv2.getTrackbarPos('y', 'track')
    w = cv2.getTrackbarPos('w', 'track')
    h = cv2.getTrackbarPos('h', 'track')
    th = cv2.getTrackbarPos('th', 'track')

    # GET VIDEO FRAME
    ret, frame = cap.read()

    # CREATE ROI
    roi = frame[y:h, x:w]

    # GRAY ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # CREATE THRESHOLD
    ret, threshold = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY_INV)

    # CREATE KERNEL
    kernel = np.ones((5, 5), np.uint8)

    # DILATE
    dilate = cv2.dilate(threshold, kernel, iterations=1)

    # COUNTOURS
    contours, h = cv2.findContours(
        dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        test_cells = []
        area = cv2.contourArea(cnt)
        try:
            if area > 300:
                x2, y2, w2, h2 = cv2.boundingRect(cnt)
                offs = 10
                crop = dilate[y2-offs:y2+offs+h2, x2-offs:w2+x2+offs]
                if crop.shape[0] > crop.shape[1]:
                    newW = (crop.shape[1]*20)/crop.shape[0]
                    crop = cv2.resize(crop, (int(newW), 20))
                else:
                    newH = (crop.shape[0]*20)/crop.shape[1]
                    crop = cv2.resize(crop, (20, int(newH)))

                cv2.rectangle(roi, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)

                height, width = crop.shape
                x3 = height if height > width else width
                y3 = height if height > width else width

                square = np.zeros((x3, y3), np.uint8)
                square[int((y3-height)/2):int(y3-(y3-height)/2),
                       int((x3-width)/2):int(x3-(x3-width)/2)] = crop

                test_cells.append(square.flatten())
                test_cells = np.array(test_cells, dtype=np.float32)

                ret, result, neighbours, dist = knn.findNearest(test_cells, k=1)
                cv2.putText(roi,str(int(result[0][0])),(x2,y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)

        except:
            pass

    # SHOW ROI
    cv2.imshow('dilate', dilate)

    # SHOW FRAME
    cv2.imshow('frame', frame)

    # STOP
    if cv2.waitKey(1) == ord('q'):
        break

# CACHE FLUSH
cap.realease()

# DESTROY ALL WINDOWS
cv2.destroyAllWindows()
