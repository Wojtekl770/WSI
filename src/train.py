import os
from face_recognition import FaceRecognizer

def main():
    # Inicjalizacja
    recognizer = FaceRecognizer()
    
    # Ścieżki względem lokalizacji skryptu
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '../data/known_faces')
    models_path = os.path.join(script_dir, '../models')
    
    if not os.path.exists(data_path):
        print(f"Błąd: Folder {data_path} nie istnieje!")
        return
        
    print("Rozpoczynam trenowanie modelu...")
    try:
        recognizer.train(data_path, models_path)
        print("Trenowanie zakończone pomyślnie!")
    except Exception as e:
        print(f"Błąd podczas trenowania: {str(e)}")

if __name__ == "__main__":
    main()