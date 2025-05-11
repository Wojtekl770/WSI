import os
import cv2
from mtcnn import MTCNN
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.utils import register_keras_serializable
from tqdm import tqdm

@register_keras_serializable()
def l2_normalize(x, axis=1):
    return tf.math.l2_normalize(x, axis=axis)

class FaceRecognizer:
    def __init__(self, input_shape=(160, 160, 3)):
        self.input_shape = input_shape
        self.model = self._build_embedding_model()
        self.classifier = None
        self.labels = []
        self.threshold = 0.6

    def _build_embedding_model(self):
        inputs = Input(shape=self.input_shape, name='input_image')
        
        # Blok konwolucyjny 1
        x = Conv2D(64, (10, 10), activation='relu', name='conv1')(inputs)
        x = BatchNormalization(name='bn1')(x)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        
        # Blok konwolucyjny 2
        x = Conv2D(128, (7, 7), activation='relu', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        
        # Blok konwolucyjny 3
        x = Conv2D(256, (4, 4), activation='relu', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = MaxPooling2D((2, 2), name='pool3')(x)
        
        # Warstwy gęste
        x = Flatten(name='flatten')(x)
        x = Dense(256, activation='relu', name='dense1')(x)
        embeddings = Dense(128, name='embeddings')(x)
        
        # Normalizacja L2
        l2_norm = Lambda(l2_normalize, name='l2_norm')(embeddings)
        
        return Model(inputs, l2_norm, name='FaceEmbeddingModel')

    def preprocess_face(self, face_image):
        """
        Przetwarza obraz twarzy przed ekstrakcją cech
        :param face_image: obraz twarzy (numpy array)
        :return: przetworzony obraz
        """
        # Zmiana rozmiaru i normalizacja
        face = cv2.resize(face_image, (self.input_shape[0], self.input_shape[1]))
        face = (face - 127.5) / 128.0  # Normalizacja do zakresu [-1, 1]
        return face

    def get_embedding(self, face_image):
        """
        Generuje embedding (wektor cech) dla twarzy
        :param face_image: obraz twarzy (numpy array)
        :return: wektor cech (128-wymiarowy)
        """
        face = self.preprocess_face(face_image)
        face = np.expand_dims(face, axis=0)  # Dodanie wymiaru batcha
        return self.model.predict(face, verbose=0)[0]

    def train(self, dataset_path, save_path='models'):
        """
        Trenuje klasyfikator na podstawie zbioru danych
        :param dataset_path: ścieżka do folderu ze zdjęciami osób
        :param save_path: gdzie zapisać wytrenowany model
        """
        X, y = [], []

        # Przetwarzanie zdjęć każdej osoby
        for person_name in os.listdir(dataset_path):
            person_dir = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_dir):
                continue
                
            print(f"Przetwarzanie osoby: {person_name}")
            
            for img_name in tqdm(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_name)
                
                try:
                    # Wczytanie i detekcja twarzy
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    faces = MTCNN().detect_faces(img)
                    
                    if faces:
                        x, y_coord, w, h = faces[0]['box']
                        face = img[y_coord:y_coord+h, x:x+w]
                        
                        # Ekstrakcja cech
                        embedding = self.get_embedding(face)
                        X.append(embedding)
                        y.append(person_name)
                except Exception as e:
                    print(f"Błąd przetwarzania {img_path}: {str(e)}")
        
        # Trenowanie klasyfikatora KNN
        self.classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        self.classifier.fit(X, y)
        self.labels = list(set(y))
        
        # Zapis modelu
        self.save_model(save_path)

    def save_model(self, save_path='models'):
        """
        Zapisuje model do plików
        :param save_path: folder docelowy
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        # Zapis modelu ekstrakcji cech
        self.model.save(os.path.join(save_path, 'embedding_model.h5'))
        
        # Zapis klasyfikatora i etykiet
        with open(os.path.join(save_path, 'classifier.pkl'), 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'labels': self.labels,
                'threshold': self.threshold
            }, f)

    def load_model(self, model_path='models'):
        """
        Ładuje zapisany model
        :param model_path: folder z modelami
        """
        # Ładowanie modelu ekstrakcji cech
        self.model = load_model(os.path.join(model_path, 'embedding_model.h5'))
        
        # Ładowanie klasyfikatora i etykiet
        with open(os.path.join(model_path, 'classifier.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.labels = data['labels']
            self.threshold = data.get('threshold', 0.6)

    def recognize(self, face_image):
        """
        Rozpoznaje osobę na obrazie twarzy
        :param face_image: obraz twarzy (numpy array)
        :return: (nazwa osoby, pewność, wektor cech)
        """
        if self.classifier is None:
            raise ValueError("Model nie został wytrenowany lub załadowany")
        
        embedding = self.get_embedding(face_image)
        
        # Sprawdzenie czy mamy jakiekolwiek etykiety
        if len(self.labels) == 0:
            return "Nieznana osoba", 1.0, embedding
        
        try:
            distances, indices = self.classifier.kneighbors([embedding], n_neighbors=1)
            
            # Jeśli odległość jest zbyt duża, twarz jest nieznana
            if distances[0][0] > self.threshold:
                return "Nieznana osoba", distances[0][0], embedding
            
            # Bezpieczne pobranie etykiety
            pred_idx = self.classifier.predict([embedding])[0]
            return pred_idx, distances[0][0], embedding
            
        except Exception as e:
            print(f"Błąd podczas klasyfikacji: {str(e)}")
            return "Nieznana osoba", 1.0, embedding

    def add_new_face(self, face_image, person_name, save_path='models'):
        """
        Dodaje nową twarz do systemu
        :param face_image: obraz twarzy
        :param person_name: nazwa osoby
        :param save_path: gdzie zapisać zaktualizowany model
        """
        if self.classifier is None:
            self.classifier = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
            self.labels = []
        
        embedding = self.get_embedding(face_image)
        
        # Aktualizacja danych treningowych
        X = self.classifier._fit_X.tolist() if hasattr(self.classifier, '_fit_X') else []
        y = self.classifier._y.tolist() if hasattr(self.classifier, '_y') else []
        
        X.append(embedding)
        y.append(person_name)
        
        # Ponowne trenowanie
        self.classifier.fit(X, y)
        self.labels = list(set(y))
        
        # Zapis zaktualizowanego modelu
        self.save_model(save_path)