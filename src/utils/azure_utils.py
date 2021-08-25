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
        self.container = ContainerClient.from_connection_string(connection_string, container_name)
        self.client = service_client.get_container_client(container_name)
        self.connection_string = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        self.container_name = sys.argv[1]

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
        blob_list = self.container.list_blobs()
        files = []
        for blob in blob_list:
            files.append(blob.name)

        return files

    def create_csv(self):
        '''
        create a csv file from blobs in the container
        '''
        fx = self.ls_files()
        hotspot_paths = ["https://www."+str(self.connection_string.split(";")[1]).split("=")[1]+".blob.core.windows.net/"+self.container_name+"/"+x for x in fx]
        hotspot_ids = [x.split(".")[0] if "json" in x else x.split("_")[0] for x in fx]
        band_type = [x.split(".")[1] if "json" in x else x.split("_")[1].split(".")[0] for x in fx]

        dictionary = {'hotspot_id': hotspot_ids, 'path': hotspot_paths, 'band_type': band_type}  
        dataframe = pd.DataFrame(dictionary) 
        
        return dataframe.to_csv("out.csv", index=False)