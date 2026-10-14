# download_yolov9_files.py
import os
import urllib.request
import sys

# Define direct paths to required files
BASE_DIR = "D:/Adobe Apps/WINTER SEM 2025-2026/VS Code/DSN/Exhibition 1/"
WEIGHTS_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt"
CFG_URL = "https://raw.githubusercontent.com/WongKinYiu/yolov9/main/models/detect/yolov9-c.yaml"
COCO_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Alternative model options (uncomment if you want different versions)
# YOLOv9-e (larger, more accurate): "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt"
# YOLOv9-t (tiny, faster): "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-t.pt"

def download_file(url, destination):
    """Download a file with progress indicator"""
    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        print(f"Downloading {os.path.basename(destination)}...")
        
        # Download the file with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(url, destination, reporthook)
        print(f"\nDownloaded {os.path.basename(destination)} successfully!")
        return True
        
    except Exception as e:
        print(f"\nError downloading {os.path.basename(destination)}: {str(e)}")
        return False

def check_existing_files():
    """Check which required files already exist"""
    existing_files = []
    missing_files = []
    
    files_to_check = {
        'yolov9-c.pt': WEIGHTS_URL,
        'yolov9-c.yaml': CFG_URL,
        'coco.names': COCO_URL
    }
    
    for file_name, file_url in files_to_check.items():
        file_path = os.path.join(BASE_DIR, file_name)
        if os.path.exists(file_path):
            existing_files.append(file_name)
            print(f"✓ {file_name} already exists")
        else:
            missing_files.append((file_name, file_url))
            print(f"✗ {file_name} is missing")
    
    return existing_files, missing_files

def main():
    print("Checking for required YOLOv9 files...")
    print(f"Target directory: {BASE_DIR}")
    print("-" * 50)
    
    existing_files, missing_files = check_existing_files()
    
    if not missing_files:
        print("\nAll required files are already present!")
        print("You can now run your YOLOv9 traffic violation detection script.")
        return
    
    print(f"\nNeed to download {len(missing_files)} files:")
    for file_name, _ in missing_files:
        print(f"  - {file_name}")
    
    # Show file sizes for user awareness (YOLOv9 weights are larger)
    print("\nNote: YOLOv9-c weights file is approximately 76MB")
    
    response = input("\nDo you want to download the missing files? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print("\nStarting download process...")
    print("-" * 50)
    
    success_count = 0
    for file_name, file_url in missing_files:
        file_path = os.path.join(BASE_DIR, file_name)
        if download_file(file_url, file_path):
            success_count += 1
    
    print("-" * 50)
    print(f"Download completed: {success_count}/{len(missing_files)} files successfully downloaded")
    
    if success_count == len(missing_files):
        print("All files downloaded successfully! You can now run your YOLOv9 traffic violation detection script.")
        print("\nNote: YOLOv9 uses .pt (PyTorch) format instead of .weights, so your detection code will need to be updated accordingly.")
    else:
        print("Some files failed to download. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()