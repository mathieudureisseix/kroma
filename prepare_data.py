import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 1. Charger le fichier CSV
csv_path = "annotations.csv"  # Remplacez par le chemin vers votre fichier CSV
data = pd.read_csv(csv_path)

# 2. Mapper les classes à des entiers
# Exemple d'encodage des classes (vous pouvez personnaliser cela)
class_mapping = {"claire": 0, "moyenne": 1, "foncée": 2}
data["class_encoded"] = data["class"].map(class_mapping)

# 3. Diviser les données en ensembles (train, val, test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)


# 4. Prétraitement des images
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Redimensionne et normalise une image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalisation
        return img_array
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return None


# 5. Appliquer le prétraitement et sauvegarder les données préparées
def prepare_dataset(dataframe, image_column, label_column, output_dir):
    """
    Prépare les images et les étiquettes pour l'entraînement.
    """
    images = []
    labels = []
    os.makedirs(output_dir, exist_ok=True)

    for _, row in dataframe.iterrows():
        image_path = row[image_column]
        label = row[label_column]

        # Prétraitement de l'image
        processed_image = preprocess_image(image_path)
        if processed_image is not None:
            images.append(processed_image)
            labels.append(label)

    # Convertir en arrays numpy
    images = np.array(images)
    labels = np.array(labels)

    # Sauvegarder les fichiers
    np.save(os.path.join(output_dir, "images.npy"), images)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    print(f"Dataset préparé et sauvegardé dans {output_dir}")


# Préparer les ensembles train, val et test
prepare_dataset(train_data, "path_rgb_original", "class_encoded", "dataset/train")
prepare_dataset(val_data, "path_rgb_original", "class_encoded", "dataset/val")
prepare_dataset(test_data, "path_rgb_original", "class_encoded", "dataset/test")

print("Préparation terminée !")