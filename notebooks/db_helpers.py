# db_helpers.py
import sqlite3
from fontTools.ttLib import TTFont
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
from pathlib import Path
import numpy as np
import threading
import time
from contextlib import contextmanager
from config.settings import DATABASE, DEBUG


# Thread-local storage for database connections
_thread_local = threading.local()

@contextmanager
def get_db_connection(db_path=DATABASE['path']):
    """Context manager for thread-safe database connections"""
    if not hasattr(_thread_local, "conn"):
        _thread_local.conn = sqlite3.connect(db_path)
        # Enable WAL mode for better concurrency
        _thread_local.conn.execute("PRAGMA journal_mode=WAL")
        _thread_local.conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
    
    yield _thread_local.conn
    
    # Connection remains open for thread reuse
    # Will be closed when thread ends

def init_db(db_path=DATABASE['path']):
    """Initialize database with proper settings"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        
        # Enable WAL mode explicitly (though already set in connection)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        
        c.execute('''CREATE TABLE IF NOT EXISTS fonts
                     (id INTEGER PRIMARY KEY,
                      family TEXT,
                      style TEXT,
                      weight INTEGER,
                      file_path TEXT UNIQUE)''')
                     
        c.execute('''CREATE TABLE IF NOT EXISTS characters
                     (font_id INTEGER,
                      char TEXT,
                      features_blob BLOB,
                      UNIQUE(font_id, char),
                      FOREIGN KEY(font_id) REFERENCES fonts(id))''')
        
        conn.commit()
    
    return True

def PILImage_to_CV2(pil_image):
    '''Convert PIL Image → OpenCV (BGR format)'''
    img_cv = np.array(pil_image)  # Shape: (H, W, 3) for RGB
    img_cv = cv2.resize(img_cv, (64, 64))  # Resize to 64x64

    
    return img_cv  # Returns a NumPy array (BGR format)


def add_font(font_path,conn, db_path=DATABASE['path'], max_retries=3):
    """Add font to database with retry logic for locking issues"""
    for attempt in range(max_retries):
        try:
                return _add_font_transaction(conn, font_path)
        except Exception as e:
            if "locked" in str(e) and attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                if DEBUG: print(f"Retrying after {0.5 * (attempt + 1)} seconds for {font_path}...")
                continue
            else:
                print(f"Error processing {font_path}: {str(e)} in add_font")
                raise

def _add_font_transaction(conn, font_path):
    """Core font addition logic within a transaction"""
    font = TTFont(font_path)
    name = font['name'].getDebugName(1)
    style = font['name'].getDebugName(2)
    weight = font['OS/2'].usWeightClass
    c = conn.cursor()
    
    # Check if font already exists
    c.execute('SELECT id FROM fonts WHERE file_path=?', (font_path,))
    existing = c.fetchone()
    if existing:
        return existing[0]  # Return existing font ID

    # Insert font record
    c.execute('INSERT INTO fonts (family, style, weight, file_path) VALUES (?,?,?,?)',
              (name, style, weight, font_path))
    font_id = c.lastrowid
    
    # Process characters in batches
    cmap = font.getBestCmap()
    if DEBUG: print(cmap)

    batch_size = 100
    char_items = [(code, chr(code)) for code in cmap]
    for i in range(0, len(char_items), batch_size):
        batch = char_items[i:i+batch_size]
        char_data = []
        
        for code, char in batch:
            try:
                img = generate_character_image(font_path, char)
                processed = preprocess_character(img)
                from pipeline_helpers import get_cnn_features
                features = get_cnn_features(processed)
                
                char_data.append((
                    font_id, 
                    char, 
                    features.tobytes()
                ))
            except Exception as e:
                print(f"Error processing character {char}: {str(e)} in add_font_transaction")
                continue
        
        # Insert batch
        if char_data:
            c.executemany('''INSERT OR IGNORE INTO characters 
                          (font_id, char, features_blob) 
                          VALUES (?,?,?)''', char_data)
            conn.commit()
            print(f"Added {font_id}"  )
    return True

def generate_character_image(font_path, char, size=256, font_size=200):
    """Generate a clean image of a single character"""
    try:
        # Create blank white image
        img = Image.new('L', (size, size), color=255)
        draw = ImageDraw.Draw(img)
        
        # Load font (handle both TTF and OTF)
        pil_font = ImageFont.truetype(font_path, size=font_size)
        
        # Get character dimensions
        bbox = draw.textbbox((0, 0), char, font=pil_font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        # Center character
        x = (size - w) // 2 - bbox[0]
        y = (size - h) // 2 - bbox[1]
        draw.text((x, y), char, font=pil_font, fill=0)
        return img
        
    except Exception as e:
        print(f"Error generating {char}: {str(e)}")
        # Return blank image if character can't be rendered
        return Image.new('L', (size, size), color=255)


def preprocess_character(img, target_size=128):
    """Process character image for feature extraction"""
    # Convert to numpy array
    img = np.array(img)
    
    # Threshold to binary (black/white)
    _, binary = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours and crop to character
    cnt, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnt:
        all_points = np.concatenate(cnt)
        x,y,w,h = cv2.boundingRect(all_points)
        img = img[y:y+h, x:x+w]
        from pipeline_helpers import pad_and_resize_char
        img = 1.0 - (img.astype(np.float32) / 255.0)
        img = pad_and_resize_char(img, target_size)
        img = cv2.resize(img, (target_size,target_size))  # Resize for CNN
    else: img = np.zeros((target_size, target_size))
    return img

def get_font_id(family=None, style=None, weight=None, file_path=None, db_path='../db/fonts.db'):
    """
    Get font ID based on font characteristics or file path.
    
    Args:
        family: Font family name
        style: Font style (e.g., 'normal', 'italic')
        weight: Font weight (e.g., 400, 700)
        file_path: Path to font file
        db_path: Path to database
        
    Returns:
        Font ID if found, None otherwise
    """
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        
        if file_path:
            # Search by file path (which is UNIQUE in the table)
            c.execute('SELECT id FROM fonts WHERE file_path = ?', (file_path,))
        else:
            # Search by font characteristics
            query = 'SELECT id FROM fonts WHERE family = ?'
            params = [family]
            
            if style is not None:
                query += ' AND style = ?'
                params.append(style)
            if weight is not None:
                query += ' AND weight = ?'
                params.append(weight)
                
            c.execute(query, params)
        
        result = c.fetchone()
        return result[0] if result else None
    

def get_fontinfo_from_id(font_id, db_path=DATABASE['path']):
    """
    Get all font information for a given font ID.
    
    Args:
        font_id: The ID of the font to look up
        db_path: Path to the database file
        
    Returns:
        A dictionary containing all font information if found, None otherwise
    """
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        c.execute('''SELECT id, family, style, weight, file_path 
                     FROM fonts 
                     WHERE id = ?''', (font_id,))
        
        result = c.fetchone()
        if result:
            return {
                'id': result[0],
                'family': result[1],
                'style': result[2],
                'weight': result[3],
                'file_path': result[4]
            }
        return None