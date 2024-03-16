import os
import sys
import shutil
import parameters as p

def get_parent_directory(path):
    directories = path.split("/")
    parent_directories = []
    current_path = ""
    for directory in directories:
        current_path += directory + "/"
        parent_directories.append(current_path.rstrip("/"))
    return parent_directories[0]

def delete_folders(folders):
    
    for folder in folders:
        folder = os.path.dirname(folder)
        try:
            shutil.rmtree(folder)
            print(f"{folder} removed successfully.")
        except FileNotFoundError:
            print(f"{folder} doesn't exist.")
        except Exception as e:
            print(f"An error occurred while deleting {folder}: {e}")

def clear_empty_folders(folders):
    for folder in folders:
        
        path = get_parent_directory(folder)

        for root, dirs, files in os.walk(path, topdown=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if not os.listdir(folder_path):
                    try:
                        os.rmdir(folder_path)
                        print(f"Empty foler {folder_path} deleted.")
                    except OSError as e:
                        print(f"An error occurred while deleting {folder_path}: {e}")
            
def delete_files(files):
    for file in files:
        try:
            os.remove(file)
            print(f"{file} removed successfully.")
        except FileNotFoundError:
            print(f"{file} doesn't exist.")
        except Exception as e:
            print(f"An error occurred while deleting {file}: {e}")

def main():
    args = sys.argv[1:]

    if not args:
        print("Usage: python clear.py all")
        print("       python clear.py checkpoint")
        print("       python clear.py data")
        print("       python clear.py previews")

        sys.exit(1)

        
    match args[0]:
        case 'all':
            confirm = input("Are you sure you want to delete all temporary files and folders ? (yes/no): ").lower()
            if confirm == 'yes':
                delete_folders([p.checkpoint_path, p.train_processed_path, p.test_processed_path, p.train_images_path, p.train_masks_path, p.test_images_path, p.test_masks_path])
                clear_empty_folders([p.checkpoint_path, p.train_processed_path, p.test_processed_path, p.train_images_path, p.train_masks_path, p.test_images_path, p.test_masks_path])
                delete_files(['loss_preview.png'])
                delete_files(['preview.png'])
            else:
                print("Deletion cancelled.")
        case 'checkpoint':
            confirm = input("Are you sure you want to delete checkpoint folder? (yes/no): ").lower()
            if confirm == 'yes':
                delete_folders([p.checkpoint_path])
                clear_empty_folders([p.checkpoint_path])
            else:
                print("Deletion cancelled.")
        case 'data':
            confirm = input("Are you sure you want to delete data folders? (yes/no): ").lower()
            if confirm == 'yes':
                delete_folders([p.train_processed_path, p.test_processed_path])
                clear_empty_folders([p.train_processed_path, p.test_processed_path])
            else:
                print("Deletion cancelled.")
        case 'previews':
            confirm = input("Are you sure you want to delete preview files? (yes/no): ").lower()
            if confirm == 'yes':
                delete_files(['loss_preview.png'])
                delete_files(['preview.png'])
            else:
                print("Deletion cancelled.")


if __name__ == "__main__":
    main()
