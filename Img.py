import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import matplotlib.pyplot as plt

# Initialize the FaceNet model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing transform for images
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Function to load and encode faces using FaceNet
def load_and_encode_images(image_folder):
    image_encodings = []
    image_paths = []

    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        image = Image.open(file_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        # Get face embeddings using FaceNet
        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy().flatten()
        
        image_encodings.append(embedding)
        image_paths.append(file_path)

    return np.array(image_encodings), image_paths

# Function to cluster faces using Agglomerative Clustering
def cluster_faces(image_encodings, n_clusters=None, distance_threshold=1.0):
    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters, 
        affinity='euclidean', 
        linkage='ward', 
        distance_threshold=distance_threshold
    )
    labels = clustering_model.fit_predict(image_encodings)
    return labels

# Function to create symbolic links for clustered images
def create_symlinks(image_paths, labels, output_folder):
    clustered_images = defaultdict(list)

    for image_path, label in zip(image_paths, labels):
        clustered_images[label].append(image_path)

    for label, images in clustered_images.items():
        label_folder = os.path.join(output_folder, f"cluster_{label}")
        os.makedirs(label_folder, exist_ok=True)

        for image_path in images:
            image_name = os.path.basename(image_path)
            symlink_path = os.path.join(label_folder, image_name)
            
            # Create symbolic link pointing to the original image
            if not os.path.exists(symlink_path):
                os.symlink(image_path, symlink_path)

# Function to display the clusters with matplotlib
def display_clusters(image_paths, labels):
    clustered_images = defaultdict(list)
    
    for image_path, label in zip(image_paths, labels):
        clustered_images[label].append(image_path)
    
    for label, images in clustered_images.items():
        print(f"Cluster {label}:")
        fig, axes = plt.subplots(1, min(5, len(images)), figsize=(20, 4))
        for img_path, ax in zip(images, axes):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis('off')
        plt.show()

if __name__ == "__main__":
    # Folder containing images
    image_folder = "C:\testimgsinput"

    # Folder to save symbolic links
    output_folder = "C:\testimgoutput"

    # Load and encode images
    image_encodings, image_paths = load_and_encode_images(image_folder)

    # Cluster faces using Agglomerative Clustering
    labels = cluster_faces(image_encodings, distance_threshold=0.7)

    # Create symbolic links for clustered images
    create_symlinks(image_paths, labels, output_folder)

    # Display the clusters
    display_clusters(image_paths, labels)

    print(f"Clustering complete. Check '{output_folder}' for symbolic links to the images.")