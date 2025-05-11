import cv2
import os
from face_detection import FaceDetector
from face_recognition import FaceRecognizer

def main():
    # Inicjalizacja
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    # Ścieżki względem lokalizacji skryptu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(script_dir, '../models')
    test_image_path = os.path.join(script_dir, '../data/unknown_faces/test2.jpg')
    
    # Sprawdzenie istnienia modelu
    if not os.path.exists(os.path.join(models_path, 'embedding_model.h5')):
        print("Błąd: Brak wytrenowanego modelu! Najpierw uruchom train.py")
        return
    
    print("Ładowanie modelu...")
    try:
        recognizer.load_model(models_path)
    except Exception as e:
        print(f"Błąd ładowania modelu: {str(e)}")
        return
    
    if not os.path.exists(test_image_path):
        print(f"Błąd: Plik {test_image_path} nie istnieje!")
        return
    
    print("Rozpoznawanie twarzy...")
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            raise ValueError("Nie można wczytać obrazu")
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(test_image_path)
        
        if not faces:
            print("Nie wykryto twarzy na zdjęciu!")
            return
        
        for face_data in faces:
            face = face_data['face']
            name, confidence, _ = recognizer.recognize(face)
            print(f"Rozpoznano: {name} (pewność: {confidence:.2f})")
            
            x, y, w, h = face_data['box']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f"{name} ({confidence:.2f})", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Wynik", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Błąd podczas rozpoznawania: {str(e)}")

if __name__ == "__main__":
    main()