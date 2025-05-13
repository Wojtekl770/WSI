# Rozpoznawanie Twarzy z Wykorzystaniem Sztucznej Inteligencji

Projekt umożliwia rozpoznawanie twarzy na podstawie zdjęć z wykorzystaniem nowoczesnych algorytmów głębokiego uczenia. System uczy się na podstawie bazy zdjęć zawierającej ponad 100 przykładów dla różnych osób, a następnie testuje skuteczność rozpoznawania z uwzględnieniem pewności klasyfikacji.

## ✨ Opis działania

Projekt zawiera trzy niezależne silniki rozpoznawania twarzy:

### 1. FaceRecognitionEngine
- **Detekcja:** klasyczny algorytm HOG (Histogram of Oriented Gradients)
- **Rozpoznawanie:** model oparty na bibliotece `dlib`, używający sieci ResNet
- **Podejście:** klasyczne metody + sieci neuronowe
- **Proces:**
  1. Detekcja twarzy
  2. Wydobycie punktów charakterystycznych (68 landmarks)
  3. Generowanie 128-wymiarowych wektorów cech (embeddings)
  4. Porównanie z bazą przy użyciu metryki euklidesowej

### 2. TensorflowFacenetEngine
- **Detekcja:** MTCNN (Multi-task Cascaded Convolutional Networks)
- **Rozpoznawanie:** model FaceNet wytrenowany z wykorzystaniem triplet loss
- **Technologie:** TensorFlow, sieci konwolucyjne (CNN)
- **Proces:**
  1. Detekcja twarzy (P-Net, R-Net, O-Net)
  2. Normalizacja twarzy
  3. Generowanie 512-wymiarowych osadzeń
  4. Porównanie z bazą twarzy

### 3. DeepFaceEngine
- **Detekcja:** różne backendy: MTCNN, RetinaFace, OpenCV
- **Rozpoznawanie:** różne modele: VGG-Face, ArcFace, FaceNet
- **Technologie:** DeepFace, CNN
- **Proces:**
  1. Detekcja
  2. Normalizacja
  3. Tworzenie wariantów twarzy
  4. Generowanie embeddings
  5. Porównanie z bazą (euklidesowa, kosinusowa)
  6. Śledzenie twarzy między klatkami
  7. Inteligentne zarządzanie bazą danych

## 🧠 Technologie

- Python
- TensorFlow
- Keras
- dlib
- face_recognition
- DeepFace
- OpenCV
- MTCNN
- ResNet, FaceNet, VGG-Face, ArcFace

## 📁 Struktura projektu
face_recognition/
├── src/
│ ├── main.py

│ ├── train.py
│ ├── test.py
│ ├── models/
│ └── ...
├── data/
│ ├── lewandowski/
│ ├── obama/
│ └── wojtek/
└── README.md


## 🧪 Etapy działania

1. **Trenowanie modelu**: uruchom `train.py`, aby przygotować model na podstawie danych treningowych.
2. **Testowanie**: uruchom `test.py`, aby przetestować system na zdjęciach testowych.
3. **Rozpoznawanie twarzy**: program analizuje obraz, wykrywa twarze i przypisuje tożsamość, jeśli wynik przekracza automatycznie dobrany próg pewności.

## ⚙️ Wymagania

- Python 3.10+
- Instalacja zależności:
  ```bash
  pip install -r requirements.txt
