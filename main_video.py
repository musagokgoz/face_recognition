import cv2
from simple_facerec import SimpleFacerec

# kendi yazdıgımız classı tanımlıyoruz
sfr = SimpleFacerec()
# aynı dizindeki images dosyasındaki fotorafları işaret ettik
sfr.load_encoding_images("images/")

# kendi kameramızı acıyoruz
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read() 
    # aynaladık yoksa video da tersliyor
    frame = cv2.flip(frame,1)
    
    # yuzleri bulmak için
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    #yuzlerı algıladık ve kare ıcıne aldık cıkan pencereye frame dedik
    cv2.imshow("Frame", frame)
    # acılan pencere kalsın ıstedık
    key = cv2.waitKey(1)
    if key == 27: # esc tusu basılırsa kır cık
        break

cap.release()
cv2.destroyAllWindows()
