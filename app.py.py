import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# Charger le modèle DNN pour la détection de visages
modelFile = "options/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "options/deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Détection de visage avec DNN
def detect_face_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            return image[startY:endY, startX:endX]
    return None

# Prétraitement des images
def preprocess_image(image, image_size=(64, 64)):
    face_img = detect_face_dnn(image)
    if face_img is not None:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, image_size)
        return face_img
    return None

# Extraction des features HOG
def extract_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block, block_norm='L2-Hys')
    return hog_features

# Chargement des images et extraction des features
def load_images_with_combined_features(folder, image_size=(64, 64)):
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for label_name in os.listdir(folder):
        label_path = os.path.join(folder, label_name)
        if os.path.isdir(label_path):
            label_map[current_label] = label_name
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = preprocess_image(cv2.imread(img_path), image_size)
                if img is not None:
                    combined_features = extract_hog_features(img)
                    images.append(combined_features)
                    labels.append(current_label)
            current_label += 1

    return np.array(images), np.array(labels), label_map

# Évaluation d'une image individuelle
def evaluate_frame_with_face_detection(frame, model, scaler, label_map, image_size=(64, 64), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    img = preprocess_image(frame, image_size)
    if img is not None:
        hog_features = extract_hog_features(img, orientations, pixels_per_cell, cells_per_block)
        hog_features = scaler.transform([hog_features])
        prediction = model.predict(hog_features)
        class_name = label_map[prediction[0]]
        return class_name
    else:
        return None

# Charger les données et préparer les ensembles d'entraînement et de test
dataset_path = './dataset'
images, labels, label_map = load_images_with_combined_features(dataset_path)

# Division des données et normalisation
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraîner un modèle SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred_svm = svm_model.predict(X_test)
print(f"Accuracy (SVM): {accuracy_score(y_test, y_pred_svm)}")
print("Classification Report (SVM):\n", classification_report(y_test, y_pred_svm, target_names=list(label_map.values())))

# Configuration de la capture vidéo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la caméra.")
else:
    print("Appuyez sur 'q' pour quitter la capture vidéo.")

# Boucle pour capturer les images de la caméra
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le cadre de la caméra.")
        break

    # Évaluer l'image capturée
    predicted_class = evaluate_frame_with_face_detection(frame, svm_model, scaler, label_map)
    
    # Afficher le résultat sur le cadre capturé
    if predicted_class is not None:
        cv2.putText(frame, f"Predicted: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Afficher le cadre
    cv2.imshow('Face Detection', frame)
    
    # Quitter la capture vidéo si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la capture vidéo et fermer toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()
#Ce code capture des images en temps réel à partir de la webcam, détecte les visages, extrait les features HOG 
# et fait des prédictions en utilisant un modèle SVM formé sur vos données.
# Vous pouvez également ajuster les hyperparamètres et améliorer le modèle selon vos besoins.

from sklearn.externals import joblib

# Assurez-vous d'avoir entraîné votre modèle et scaler
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')
