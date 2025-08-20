import os

DIR = "/home/nepomuk/sflnas/DataReadOnly334/tractor_data"

def list_files_in_directory(directory):
    try:
        return os.listdir(directory)
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")
        return []

if __name__ == "__main__":
    files = list_files_in_directory(DIR)
    print("Files in directory:", files)