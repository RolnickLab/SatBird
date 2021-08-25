import os
import csv
import pandas as pd
import sys
import argparse

from azure.storage.blob import BlobServiceClient, ContainerClient

class DirectoryClient:
    '''
    Helper class for Azure file ops
    '''
    def __init__(self, connection_string, container_name):
        service_client = BlobServiceClient.from_connection_string(connection_string)
        container = ContainerClient.from_connection_string(connection_string, container_name)
        self.client = service_client.get_container_client(container_name)

    def upload_file(self, local_path, blob_name):
        '''
        upload a file to the container
        '''
        pass

    def download_file(self, blob_name, local_path):
        '''
        download a file from the container
        '''
        pass


    def ls_blob_files(self, path, recursive=False):
        '''
        List files under a path, optionally recursively
        '''
        if not path == '' and not path.endswith('/'):
            path += '/'

        blob_iter = self.client.list_blobs(name_starts_with=path)
        files = []
        for blob in blob_iter:
            relative_path = os.path.relpath(blob.name, path)
            if recursive or not '/' in relative_path:
                files.append(relative_path)
            return files

    def ls_files(self):
        '''
        list files in the container
        '''
        blob_list = container.list_blobs()
        files = []
        for blob in blob_list:
            files.append(blob.name)

        return files

    def create_csv(self):
        '''
        create a csv file from blobs in the container
        '''
        fx = self.ls_files()
        hotspot_paths = ["https://www."+str(CONNECTION_STRING.split(";")[1]).split("=")[1]+".blob.core.windows.net/"+CONTAINER_NAME+"/"+x for x in fx]
        hotspot_ids = [x.split(".")[0] if "json" in x else x.split("_")[0] for x in fx]
        band_type = [x.split(".")[1] if "json" in x else x.split("_")[1].split(".")[0] for x in fx]

        dictionary = {'hotspot_id': hotspot_ids, 'path': hotspot_paths, 'band_type': band_type}  
        dataframe = pd.DataFrame(dictionary) 
        
        return dataframe.to_csv("out.csv", index=False)

    
# main
if __name__ == '__main__':

    try:
        CONNECTION_STRING = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    except KeyError:
        print('AZURE_STORAGE_CONNECTION_STRING must be set')
        sys.exit(1)

    try:
        CONTAINER_NAME = sys.argv[1]
    except IndexError:
        print('usage: directory_interface.py CONTAINER_NAME')
        print('error: the following arguments are required: CONTAINER_NAME')
        print("\n Example: python src/load_data.py sentinel")
        sys.exit(1)



    client = DirectoryClient(CONNECTION_STRING, CONTAINER_NAME)
    container = ContainerClient.from_connection_string(CONNECTION_STRING, container_name=CONTAINER_NAME)

    f = client.ls_files()
    client.create_csv()