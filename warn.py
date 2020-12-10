#import modules
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


# grab the dimensions of the real_T_vid and then construct a blob from it
#array of faces to store face after preprocessing
#location is for storing location of face area
#prediction for storing predicted value of model
def mask_detection(real_T_vid, Net, MODEL):
    (h,w)=real_T_vid.shape[:2]
    blob=cv2.dnn.blobFromImage(real_T_vid, 1.0, (224, 224),(104.0, 177.0, 123.0))
    Net.setInput(blob)
    detections= Net.forward()
    faces=[]
    location=[]
    prediction=[]
    
    for i in range(0, detections.shape[2]):
        confidence= detections[0,0,i,2] #extract probability associated with detection
        
        if confidence>0.5:
            box= detections[0,0,i,3:7]*np.array([w,h,w,h])
            (X1,Y1,X2,Y2)=box.astype("int")
            (X1, Y1) = (max(0, X1), max(0, Y1))
            (X2, Y2) = (min(w-1, X2), min(h-1, Y2))
            
            face = real_T_vid[Y1:Y2, X1:X2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face) 
            location.append((X1,Y1,X2,Y2))
        #proceed if one or more face is detected
        if len(faces)>0:
            faces = np.array(faces, dtype="float32")
            prediction = MODEL.predict(faces, batch_size=32)
        return (location, prediction)
       
proto_path = r"deploy.prototxt"
weight_path = r"res10_300x300_ssd_iter_140000.caffemodel"
Net = cv2.dnn.readNet(proto_path, weight_path)

MODEL = load_model("mask_detector.model")

video = cv2.VideoCapture(0)
while True:
    _, real_T_vid = video.read()
    ht,wd = real_T_vid.shape[:2]
    real_T_vid = cv2.resize(real_T_vid, dsize=(600,ht))
    (location, prediction) = mask_detection(real_T_vid, Net, MODEL)
        
    for (box, pred) in zip(location, prediction):
        (X1, Y1, X2, Y2) = box
        (mask, withoutMask) = pred
                    
        label = "Mask" if mask > .75 else "No Mask"
        color = (100, 155, 0) if label == "Mask" else (0, 100, 155)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
        # To define the distance from camera, change the constant
        # value 60 in the following if condition
        if X2-X1>(60):
            # Check if prediction for without mask is 75% and above
            if withoutMask>.75:
                cv2.putText(real_T_vid,'Maintain Distance',(X2, Y2-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                # print("Warning")
        cv2.putText(real_T_vid, label, (X1, Y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.rectangle(real_T_vid, (X1, Y1), (X2, Y2), color, 2)
            
    cv2.imshow("real_T_vid", real_T_vid)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):	
        break
#close poped up window
cv2.destroyAllWindows()
video.stop()
        
