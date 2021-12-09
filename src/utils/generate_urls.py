import azure_utils
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure_utils import DirectoryClient
import os
import sys

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