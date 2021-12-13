import os

def create_barchart_folder(barchart_folder):
    """
    barchart_folder = folder where barchart images are created
    """
    # Delete directory if it exists aned create a new one
    if os.path.exists(barchart_folder) == False:
        print(" HBar Chart folder doesn't exist")

    if os.path.isdir(barchart_folder):
        print("Exists")
        shutil.rmtree(barchart_folder)
        print("Deleted")

    os.mkdir(barchart_folder)