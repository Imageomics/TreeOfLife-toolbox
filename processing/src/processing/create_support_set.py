import os
from io import BytesIO
import json
import hashlib
import requests
from urllib.parse import urlparse
import argparse

from sklearn.cluster import KMeans
import torch
import clip
import torch.nn.functional as F
from torchvision import transforms
from typing import Literal, Union
import pandas as pd
import numpy as np
from PIL import Image




def download_image(url: str, path: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image from {url}")
        return False
    with open(path, 'wb') as f:
        f.write(response.content)
    return True

def download_category_images(category_data: dict, target_dir: str):
    """
    Downloads images for a category and returns metadata about successful and failed downloads
    
    Args:
        category_data: Dictionary containing category info and URLs
        target_dir: Target directory for downloads
        
    Returns:
        tuple: (successful_downloads, failed_urls) where
            successful_downloads: list of dicts with keys ["url", "path", "category"]
            failed_urls: list of failed URLs
    """
    os.makedirs(target_dir, exist_ok=True)
    
    successful = []
    failed = []
    category_label = category_data.get("label", "")

    print(f"Downloading images for category {category_label}")
    
    for url in category_data.get("data_urls"):
        # Extract filename from URL
        # filename = os.path.basename(urlparse(url).path)
        # if not filename:
        #     filename = url.split('/')[-1]
        
        # # Ensure filename has extension
        # if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
        #     filename += '.png'

        filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
            
        save_path = os.path.join(target_dir, filename)
        abs_path = os.path.abspath(save_path)
        
        if download_image(url, save_path):
            successful.append({
                "url": url,
                "path": abs_path,
                "category": category_label
            })
        else:
            failed.append(url)
            
    return successful, failed

def get_img_embeddings(
    img: Union[str, BytesIO, np.ndarray], 
    input_type: Literal["file", "url", "array"], 
    model, 
    preprocess, 
    device
):
    """
    Generates image embeddings using CLIP for a given image input.

    Parameters:
        img: str, file-like object, or np.ndarray
            - If input_type is "file", this should be a file path or file-like object.
            - If input_type is "url", this should be a string URL.
            - If input_type is "array", this should be a numpy ndarray in HWC format (RGB).
        input_type: Literal["file", "url", "array"]
            The type of the input (local file, URL, or numpy array).
        model: torch.nn.Module
            The CLIP model loaded using clip.load().
        preprocess: Callable
            The preprocessing function returned by clip.load().
        device: torch.device
            The device on which to run the model.

    Returns:
        torch.Tensor: The image embedding tensor.
    """
    
    # Load the image based on input_type
    if input_type == "file":
        image = Image.open(img).convert("RGB")
    elif input_type == "url":
        response = requests.get(img)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch image from URL: {img}")
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif input_type == "array":
        if not isinstance(img, np.ndarray):
            raise TypeError("Expected numpy array for input_type='array'")
        image = Image.fromarray(img, mode="RGB")
    else:
        raise ValueError("Invalid input_type. Choose from 'file', 'url', or 'array'.")

    # Preprocess and embed
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    return image_features

def create_clustered_class_embeddings(support_embeddings_dict, clusters_per_class):
    """
    Args:
        support_embeddings_dict: Dict[str, List[Tensor]]
            Example: {'cat': [tensor1, tensor2, ...], 'dog': [tensor1, tensor2, ...]}
        clusters_per_class: Dict[str, int]
            Example: {'cat': 3, 'dog': 2, 'rabbit': 1}

    Returns:
        Dict[str, Tensor]
            Example: {'cat/0': Tensor, 'cat/1': Tensor, 'dog/0': Tensor, ...}
    """
    clustered_dict = {}

    for label, embeddings in support_embeddings_dict.items():
        X = torch.stack(embeddings).squeeze(1).cpu().numpy()
        k = clusters_per_class.get(label, 1)

        if len(X) < k:
            # Fallback to mean if not enough samples
            print(f"Warning: Not enough samples to form {k} clusters for '{label}'. Using mean.")
            clustered_dict[label] = torch.tensor(X).mean(dim=0)
            continue

        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        for i, center in enumerate(kmeans.cluster_centers_):
            key = f"{label}/{i}" if k > 1 else label
            clustered_dict[key] = torch.tensor(center).unsqueeze(0)

    return clustered_dict

def main(support_set_urls_path: str):

    # ============================== #
    # ---- Download Support Set ----
    # ============================== #

    with open(support_set_urls_path, 'r') as f:
        metadata = json.load(f)
    
    target_dir = metadata.get('target_dir')
    categories = metadata.get("categrories")

    all_successful = []
    all_failed = []

    for category_name, category_data in categories.items():
        category_target_dir = os.path.join(target_dir, category_name)
        successful, failed = download_category_images(category_data, category_target_dir)

        all_successful.extend(successful)
        all_failed.extend(failed)

        print(f"Downloaded {len(successful)} images for category {category_name}")
        print(f"Failed to download {len(failed)} images for category {category_name}")
    
    df_successful = pd.DataFrame(all_successful)
    df_successful.to_csv(os.path.join(target_dir, "support_set_metadata.csv"), index=False)
    print(f"\nTotal successful downloads: {len(all_successful)}")
    print(f"Total failed downloads: {len(all_failed)}")
    print(f"Failing URLs: {all_failed}")

    # ================================= #
    # ---- Create Image Embeddings ----
    # ================================= #
    # Load CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L/14@336px"
    model, preprocess = clip.load(model_name, device=device)
    print(f"Creating Image Embeddings using CLIP model {model_name}")

    # Create embeddings for each image, each category
    img_class_metadata_dict =  df_successful.groupby("category")["path"].apply(list).to_dict()
    img_class_embeddings_dict = {}

    for label, image_paths in img_class_metadata_dict.items():
        embeddings = []
        for img_path in image_paths:
            try:
                embedding = get_img_embeddings(img_path, "file", model, preprocess, device)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                embeddings.append(None)  # Mark failed cases as None
        img_class_embeddings_dict[label] = embeddings
    
    # Dictionary to store mean embeddings for each class
    class_embeddings_mean = {}

    for label, image_embeddings_list in img_class_embeddings_dict.items():
        class_embeddings_mean[label] = torch.stack(image_embeddings_list).mean(dim=0)  

    # Create K-means clusters for embeddings
    k_dict = {}
    for category_name, category_data in metadata['categrories'].items():
        # Use the label as the key
        label = category_data['label']
        k = category_data.get('k', 1)
        k_dict[label] = k
    
    class_embeddings_clustered = create_clustered_class_embeddings(
        img_class_embeddings_dict, k_dict
    )

    # Save embeddings to disk
    torch.save(
        class_embeddings_mean,
        os.path.join(target_dir, "support_set_mean_embeddings.pt") 
    )

    torch.save(
        class_embeddings_clustered,
        os.path.join(target_dir, "support_set_clustered_embeddings.pt") 
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download support set images")
    parser.add_argument("support_set_urls_path", type=str, help="Path to JSON file containing support set URLs")
    args = parser.parse_args()
    
    main(args.support_set_urls_path)