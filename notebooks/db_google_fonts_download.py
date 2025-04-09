# Download the fonts
import os
import requests
from pathlib import Path
import concurrent.futures

# Configuration
GITHUB_REPO = "google/fonts"
BRANCH = "main"
BASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/git/trees/{BRANCH}?recursive=1"
OUTPUT_DIR = Path("../db/google-fonts/")
MAX_WORKERS = 10  # Number of parallel downloads

def get_all_font_files():
    """Get all .ttf file paths from the GitHub repo"""
    print("Fetching font file list from GitHub...")
    response = requests.get(BASE_URL)
    response.raise_for_status()
    
    all_files = response.json()["tree"]
    ttf_files = [f["path"] for f in all_files if f["path"].endswith(".ttf")]
    
    print(f"Found {len(ttf_files)} .ttf files")
    return ttf_files

def download_file(file_path):
    """Download a single font file"""
    try:
        # Create output path
        rel_path = Path(file_path)
        output_path = OUTPUT_DIR / rel_path.relative_to("ofl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if output_path.exists():
            return f"Skipped (exists): {file_path}"
        
        # Download file
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{BRANCH}/{file_path}"
        response = requests.get(raw_url, stream=True)
        response.raise_for_status()
        
        # Save file
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return f"Downloaded: {file_path}"
    except Exception as e:
        return f"Failed: {file_path} - {str(e)}"

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all font files
    ttf_files = get_all_font_files()
    
    # Download files in parallel
    print("Starting downloads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_file, file_path) for file_path in ttf_files]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            print(f"\rProcessed {i+1}/{len(ttf_files)} files", end="")
            result = future.result()
            # Uncomment to see detailed download results
            # print(result)
    
    print("\nAll downloads completed!")

if __name__ == "__main__":
    main()