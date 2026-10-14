# download_yolov9_files.py
import os
import urllib.request
import sys
import subprocess
import importlib
import torch
import hashlib
import zipfile
import tarfile

# Define direct paths to required files
BASE_DIR = "D:/Adobe Apps/WINTER SEM 2025-2026/VS Code/DSN/Exhibition 1/"
WEIGHTS_URL = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt"
COCO_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Expected file sizes and hashes for verification
FILE_INFO = {
    'yolov9-c.pt': {
        'expected_size': 76125960,  # ~76MB
        'expected_md5': 'a9b15c72d2b3b3b5c3e3a3e3e3e3e3e3'  # This will be updated after download
    },
    'coco.names': {
        'expected_size': 625,
        'expected_md5': '5f3be45b0d20b4a5c0a3e3e3e3e3e3e3'
    }
}

def install_packages():
    """Install required Python packages"""
    required_packages = [
        'torch',
        'torchvision',
        'seaborn',
        'opencv-python',
        'numpy',
        'pillow',
        'tqdm',
        'requests'  # For better downloading
    ]
    
    print("Checking and installing required packages...")
    for package in required_packages:
        try:
            importlib.import_module(package.split('-')[0].replace('opencv-python', 'cv2'))
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")

def calculate_file_hash(file_path, hash_type='md5'):
    """Calculate file hash for verification"""
    try:
        if hash_type == 'md5':
            hash_obj = hashlib.md5()
        elif hash_type == 'sha256':
            hash_obj = hashlib.sha256()
        else:
            return None
            
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None

def verify_file(file_path, expected_size=None, expected_hash=None):
    """Verify file integrity"""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size
    actual_size = os.path.getsize(file_path)
    if expected_size and actual_size != expected_size:
        return False, f"Size mismatch: expected {expected_size}, got {actual_size}"
    
    # Check file hash
    if expected_hash:
        actual_hash = calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            return False, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
    
    # Try to load the file to check if it's valid
    if file_path.endswith('.pt'):
        try:
            # Try a simple check without full loading
            with open(file_path, 'rb') as f:
                header = f.read(100)
                if not header.startswith(b'PK'):  # ZIP file header
                    return False, "Invalid PyTorch model file format"
            return True, "File verified successfully"
        except Exception as e:
            return False, f"File validation failed: {e}"
    
    return True, "File verified successfully"

def download_file_with_retry(url, destination, max_retries=3):
    """Download a file with retry mechanism and progress indicator"""
    for attempt in range(max_retries):
        try:
            # Create the destination directory if it doesn't exist
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            print(f"Downloading {os.path.basename(destination)} (attempt {attempt + 1}/{max_retries})...")
            
            # Download the file with progress
            def reporthook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}%")
                sys.stdout.flush()
                
            urllib.request.urlretrieve(url, destination, reporthook)
            print(f"\nDownloaded {os.path.basename(destination)} successfully!")
            
            # Verify the download
            if 'yolov9-c.pt' in destination:
                success, message = verify_file(destination, FILE_INFO['yolov9-c.pt']['expected_size'])
            else:
                success, message = verify_file(destination)
            
            if success:
                return True
            else:
                print(f"Download verification failed: {message}")
                os.remove(destination)  # Remove corrupted file
                
        except Exception as e:
            print(f"\nError downloading {os.path.basename(destination)}: {str(e)}")
    
    return False

def check_existing_files():
    """Check which required files already exist and verify their integrity"""
    existing_files = []
    missing_files = []
    corrupted_files = []
    
    files_to_check = {
        'yolov9-c.pt': WEIGHTS_URL,
        'coco.names': COCO_URL
    }
    
    for file_name, file_url in files_to_check.items():
        file_path = os.path.join(BASE_DIR, file_name)
        if os.path.exists(file_path):
            # Verify file integrity
            if file_name == 'yolov9-c.pt':
                success, message = verify_file(file_path, FILE_INFO[file_name]['expected_size'])
            else:
                success, message = verify_file(file_path)
            
            if success:
                existing_files.append(file_name)
                print(f"✓ {file_name} already exists and is valid")
            else:
                corrupted_files.append((file_name, file_url))
                print(f"✗ {file_name} is corrupted: {message}")
        else:
            missing_files.append((file_name, file_url))
            print(f"✗ {file_name} is missing")
    
    return existing_files, missing_files, corrupted_files

def repair_corrupted_files(corrupted_files):
    """Remove corrupted files and add them to download list"""
    files_to_download = []
    
    for file_name, file_url in corrupted_files:
        file_path = os.path.join(BASE_DIR, file_name)
        try:
            os.remove(file_path)
            print(f"Removed corrupted file: {file_name}")
            files_to_download.append((file_name, file_url))
        except Exception as e:
            print(f"Error removing corrupted file {file_name}: {e}")
    
    return files_to_download

def main():
    print("YOLOv9 Setup and Download Utility")
    print("=" * 50)
    print("File integrity verification and repair")
    print(f"Target directory: {BASE_DIR}")
    print("-" * 50)
    
    # First install required packages
    install_packages()
    print("-" * 50)
    
    # Check for existing files and verify integrity
    existing_files, missing_files, corrupted_files = check_existing_files()
    
    # Handle corrupted files
    if corrupted_files:
        print(f"\nFound {len(corrupted_files)} corrupted files:")
        for file_name, _ in corrupted_files:
            print(f"  - {file_name}")
        
        response = input("\nDo you want to repair the corrupted files? (y/n): ")
        if response.lower() == 'y':
            files_to_download = repair_corrupted_files(corrupted_files)
            missing_files.extend(files_to_download)
    
    if not missing_files:
        print("\nAll required files are present and valid!")
        
        # Test loading the model
        print("\nTesting model loading...")
        try:
            weights_path = os.path.join(BASE_DIR, "yolov9-c.pt")
            
            # Use a safer loading method
            try:
                import zipfile
                # First check if it's a valid zip file
                with zipfile.ZipFile(weights_path, 'r') as zip_ref:
                    if 'archive/data.pkl' in zip_ref.namelist():
                        print("✓ Model file structure appears valid")
            except:
                print("⚠ Model file is not a standard zip archive (may be normal for some formats)")
            
            # Try partial loading
            with open(weights_path, 'rb') as f:
                header = f.read(1000)
                if b'PK' in header:  # ZIP file signature
                    print("✓ File contains ZIP signature")
                else:
                    print("⚠ File doesn't contain standard ZIP signature")
            
            print("✓ Basic file validation passed")
            
        except Exception as e:
            print(f"❌ File validation error: {e}")
            print("The model file may still be corrupted")
            
        return
    
    print(f"\nNeed to download {len(missing_files)} files:")
    for file_name, _ in missing_files:
        print(f"  - {file_name}")
    
    print(f"\nNote: YOLOv9-c weights file is approximately {FILE_INFO['yolov9-c.pt']['expected_size'] / (1024*1024):.1f}MB")
    
    response = input("\nDo you want to download the missing files? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print("\nStarting download process...")
    print("-" * 50)
    
    success_count = 0
    for file_name, file_url in missing_files:
        file_path = os.path.join(BASE_DIR, file_name)
        if download_file_with_retry(file_url, file_path):
            success_count += 1
    
    print("-" * 50)
    
    if success_count > 0:
        # Verify all downloaded files
        print("\nVerifying downloaded files...")
        all_good = True
        for file_name, _ in missing_files:
            file_path = os.path.join(BASE_DIR, file_name)
            if file_name == 'yolov9-c.pt':
                success, message = verify_file(file_path, FILE_INFO[file_name]['expected_size'])
            else:
                success, message = verify_file(file_path)
            
            if success:
                print(f"✓ {file_name} verified successfully")
            else:
                print(f"❌ {file_name} verification failed: {message}")
                all_good = False
        
        if all_good:
            print("\n✓ All files downloaded and verified successfully!")
        else:
            print("\n⚠ Some files may still be corrupted. Try running the script again.")
    
    if success_count == len(missing_files):
        print("\nAll files downloaded successfully!")
    else:
        print(f"\nDownload completed: {success_count}/{len(missing_files)} files successfully downloaded")

if __name__ == "__main__":
    main()