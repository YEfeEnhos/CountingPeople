import numpy as np
import imutils
import dlib
import cv2
from scipy.spatial import distance as dist
from collections import OrderedDict
import os

class Person:
    def __init__(self, ID, center, box):
        self.ID = ID
        self.center = [center]
        self.box = box  

class TrackingSystem:

    def __init__(self, invisibilityLimit=50, disappearanceLimit=50):
        self.nextID = 1
        self.people = OrderedDict()
        self.invisibilityDurations = OrderedDict()
        self.invisibilityLimit = invisibilityLimit
        self.disappearanceLimit = disappearanceLimit

    def createPerson(self, center, box):
        self.people[self.nextID] = {'center': center, 'box': box}
        self.invisibilityDurations[self.nextID] = 0
        self.nextID += 1

    def deletePerson(self, ID):
        del self.people[ID]
        del self.invisibilityDurations[ID]

    def localDivergence(self, old_box, new_box):
        """ Calculate local divergence based on bounding box change. """
        (ox1, oy1, ox2, oy2) = old_box
        (nx1, ny1, nx2, ny2) = new_box
        

        old_width, old_height = ox2 - ox1, oy2 - oy1
        new_width, new_height = nx2 - nx1, ny2 - ny1
        
        center_shift = dist.euclidean(
            ((ox1 + ox2) / 2, (oy1 + oy2) / 2),
            ((nx1 + nx2) / 2, (ny1 + ny2) / 2)
        )
        
        width_change = abs(new_width - old_width) / old_width
        height_change = abs(new_height - old_height) / old_height

        divergence = center_shift + width_change + height_change
        return divergence

    def update(self, regions):
        if len(regions) == 0:
            for ID in list(self.invisibilityDurations.keys()):
                self.invisibilityDurations[ID] += 1
                if self.invisibilityDurations[ID] > self.invisibilityLimit:
                    self.deletePerson(ID)
            return self.people

        newCenters = np.zeros((len(regions), 2), dtype="int")
        newBoxes = []

        for (i, (startX, startY, endX, endY)) in enumerate(regions):
            centerX = int((startX + endX) / 2.0)
            centerY = int((startY + endY) / 2.0)
            newCenters[i] = (centerX, centerY)
            newBoxes.append((startX, startY, endX, endY))

        if len(self.people) == 0:
            for i in range(0, len(newCenters)):
                self.createPerson(newCenters[i], newBoxes[i])
        else:
            personIDs = list(self.people.keys())
            originalBoxes = [data['box'] for data in self.people.values()]

            divergence_matrix = np.zeros((len(originalBoxes), len(newBoxes)), dtype="float")
            for i, old_box in enumerate(originalBoxes):
                for j, new_box in enumerate(newBoxes):
                    divergence_matrix[i, j] = self.localDivergence(old_box, new_box)

            rows = divergence_matrix.min(axis=1).argsort()
            cols = divergence_matrix.argmin(axis=1)[rows]

            matchedRows = set()
            matchedCols = set()

            for (originalRow, newRow) in zip(rows, cols):
                if originalRow in matchedRows or newRow in matchedCols:
                    continue
                if divergence_matrix[originalRow, newRow] > self.disappearanceLimit:
                    continue

                ID = personIDs[originalRow]
                self.people[ID] = {'center': newCenters[newRow], 'box': newBoxes[newRow]}
                self.invisibilityDurations[ID] = 0
                matchedRows.add(originalRow)
                matchedCols.add(newRow)

            nowMissingOriginals = set(range(0, divergence_matrix.shape[0])).difference(matchedRows)
            newComers = set(range(0, divergence_matrix.shape[1])).difference(matchedCols)

            if divergence_matrix.shape[0] >= divergence_matrix.shape[1]:
                for person in nowMissingOriginals:
                    ID = personIDs[person]
                    self.invisibilityDurations[ID] += 1
                    if self.invisibilityDurations[ID] > self.invisibilityLimit:
                        self.deletePerson(ID)
            else:
                for person in newComers:
                    self.createPerson(newCenters[person], newBoxes[person])

        return self.people


tracking_system = TrackingSystem(invisibilityLimit=40, disappearanceLimit=50)
dlibTrackers = []
people = {}
frameCount = 0
totalPersonCount = 0

videoFile = "input-video" #update to match your file name
model = "MobileNetSSD.caffemodel"
prototext = "MobileNetSSD.prototxt" #update according to your model
requiredConfidence = 0.4
recheckTime = 30

net = cv2.dnn.readNetFromCaffe(prototext, model)
video = cv2.VideoCapture(videoFile)

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

output_folder = "output_frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = imutils.resize(frame, width=500)  
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (H, W) = frame.shape[:2]

    personRegions = []

    if frameCount % recheckTime == 0:
        dlibTrackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
        foundPeople = detections[0, 0]

        for index in np.arange(0, detections.shape[2]):
            confidence = foundPeople[index, 2]

            if confidence > requiredConfidence:
                classID = int(foundPeople[index, 1])
                if classID != 15:
                    continue

                box = foundPeople[index, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                dlibTrackers.append(tracker)
    else:
        for tracker in dlibTrackers:
            tracker.update(rgb)
            pos = tracker.get_position()
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            personRegions.append((startX, startY, endX, endY))

    finalPeople = tracking_system.update(personRegions)

    for (personID, info) in finalPeople.items():
        center = info['center']
        if personID > totalPersonCount:
            totalPersonCount = personID

        detectedPerson = people.get(personID, None)

        if detectedPerson is None:
            detectedPerson = Person(personID, center, info['box'])

        people[personID] = detectedPerson

        cv2.putText(frame, str(personID), (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.circle(frame, (center[0], center[1]), 4, (255, 0, 255))

    frame_path = os.path.join(output_folder, f"frame_{frameCount:04d}.jpg")
    cv2.imwrite(frame_path, frame)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frameCount += 1

print(totalPersonCount, "people passed")
video.release()
cv2.destroyAllWindows()
