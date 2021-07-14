import os

def create_barchart_folder(barchart_folder):

    # Delete directory if it exists and create a new one
    if os.path.exists(barchart_folder) == False:
        print(" HBar Chart folder doesn't exist")

    if os.path.isdir(barchart_folder):
        print("Exists")
        shutil.rmtree(barchart_folder)
        print("Deleted")

    os.mkdir(barchart_folder)