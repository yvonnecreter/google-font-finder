import cv2
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import typing_extensions
from torchvision.models import resnet18

# Initialize CNN model
model = resnet18(pretrained=True)
model.eval()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Autocrop
    coords = cv2.findNonZero(thresh)
    x,y,w,h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    
    return cropped

def segment_characters(img):
    # Denoising
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    chars = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        char_img = img[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (64,64))  # Resize for CNN
        chars.append((x, char_img))
    
    # Sort left-to-right
    return [img for x, img in sorted(chars, key=lambda x: x[0])]

def get_cnn_features(img):
    """Simplified CNN feature extraction for testing"""
    # Mock implementation - replace with actual model
    if len(img.shape) == 2:  # Grayscale
        img = np.stack([img]*3, axis=0)  # Convert to 3 channels
    else:
        img = np.transpose(img, (2, 0, 1))  # CHW format
    
    # Simple feature extraction - replace with actual model
    features = np.concatenate([
        img.mean(axis=(1, 2)),  # Channel means
        img.std(axis=(1, 2)),   # Channel stds
        img.reshape(-1)[::100]   # Sample of pixels
    ])
    return features.astype(np.float32)

# data_helpers.py (additional functions)
def get_font_features(font_id, conn):
    c = conn.cursor()
    c.execute('''SELECT char, svg_path FROM characters 
                 WHERE font_id = ?''', (font_id,))
    features = []
    for char, svg_path in c.fetchall():
        # Load pre-generated CNN features for each character
        # (Should be precomputed during database population)
        features.append(load_features(svg_path))
    return np.mean(features, axis=0)

def find_similar_fonts(target_features, conn, top_n=5):
    c = conn.cursor()
    c.execute('SELECT id FROM fonts')
    all_fonts = [row[0] for row in c.fetchall()]
    
    similarities = []
    for font_id in all_fonts:
        font_features = get_font_features(font_id, conn)
        sim = cosine_similarity(target_features, font_features)
        similarities.append((font_id, sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

def load_features(conn, font_id, char):
    """Load precomputed features from database"""
    c = conn.cursor()
    c.execute('''SELECT features_blob FROM characters 
                 WHERE font_id=? AND char=?''', (font_id, char))
    blob = c.fetchone()[0]
    return np.frombuffer(blob, dtype=np.float32)

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def match_characters(char_img, char, db_path='../db/fonts.db'):
    """Thread-safe character matching with database connection handling"""
    features = get_cnn_features(char_img)
    print(features)
    matches = []
    
    # Use our thread-safe connection context manager
    from db_helpers import get_db_connection
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        
        # First get all potential matches
        c.execute('''SELECT font_id, features_blob FROM characters 
                     WHERE char = ?''', (char,))
        
        # Process each match
        for font_id, features_blob in c.fetchall():
            if features_blob is None:
                continue
                
            try:
                # Load and compare features
                db_features = np.frombuffer(features_blob, dtype=np.float32)
                
                # Reshape for sklearn's cosine_similarity
                features_2d = features.reshape(1, -1)
                db_features_2d = db_features.reshape(1, -1)
                
                similarity = cosine_similarity(features_2d, db_features_2d)[0][0]
                matches.append((font_id, similarity))
            except Exception as e:
                print(f"Error comparing features for font {font_id}: {str(e)}")
                continue

    return matches

def batch_match_characters(char_images, chars, db_path='../db/fonts.db'):
    """Optimized batch matching of multiple characters"""
    # Pre-compute all features first
    features_list = [get_cnn_features(img) for img in char_images]
    
    results = []
    from db_helpers import get_db_connection
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        
        for char, features in zip(chars, features_list):
            char_matches = []
            c.execute('''SELECT font_id, features_blob FROM characters 
                         WHERE char = ?''', (char,))
            
            for font_id, features_blob in c.fetchall():
                if features_blob is None:
                    continue
                    
                try:
                    db_features = np.frombuffer(features_blob, dtype=np.float32)
                    similarity = cosine_similarity(
                        features.reshape(1, -1),
                        db_features.reshape(1, -1)
                    )[0][0]
                    char_matches.append((font_id, similarity))
                except Exception as e:
                    print(f"Error processing font {font_id}: {str(e)}")
                    continue
            results.append(char_matches)
    return results

def get_cnn_features(img):
    """Modified to work with thread-safe connection handling"""
    # Convert to 3 channels if grayscale
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=2)  # HWC format
    
    # Normalize and convert to tensor
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1)  # CHW format
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim
    
    # Get features - using our thread-safe model
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().flatten()