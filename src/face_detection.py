from mtcnn import MTCNN
import cv2

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()  # Usuń min_face_size
    
    def detect_faces(self, image_path):
        """Wykrywa twarze i zwraca wycięte obszary twarzy oraz punkty charakterystyczne"""
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(image)
        
        faces = []
        for res in results:
            x, y, w, h = res['box']
            # Zabezpieczenie przed ujemnymi współrzędnymi
            x, y = max(0, x), max(0, y)
            face = image[y:y+h, x:x+w]
            keypoints = res['keypoints']
            faces.append({
                'face': face,
                'keypoints': keypoints,
                'box': (x, y, w, h)
            })
        return faces