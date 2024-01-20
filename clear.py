import os
import sys
import shutil
import parameters as p

def delete_folders(folders):
    for folder in folders:
        try:
            shutil.rmtree(folder)
            print(f"{folder} removed successfully.")
        except FileNotFoundError:
            print(f"{folder} doesn't exist.")
        except Exception as e:
            print(f"An error occurred while deleting {folder}: {e}")
            
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
        print("       python clear.py result")
        print("       python clear.py checkpoint")
        print("       python clear.py data")
        print("       python clear.py stat")

        sys.exit(1)

        
    match args[0]:
        case 'all':
            delete_folders([p.results_path, p.checkpoint_path, p.train_processed_path, p.test_processed_path])
            delete_files(['loss.png'])      
        case 'result':
            delete_folders([p.results_path])
        case 'checkpoint':
            delete_folders([p.checkpoint_path])
        case 'data':
            delete_folders([p.train_processed_path, p.test_processed_path,])   
        case 'stat':
            delete_files(['loss.png'])      


if __name__ == "__main__":
    main()
