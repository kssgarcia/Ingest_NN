# %%
import os
import json
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import numpy as np

s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Load the JSON data from the file
with open('data.json', 'r') as file:
    data = json.load(file)

# Iterate through each entry in the JSON data
for entry in data:
    # Access the 'data' and 'annotations' fields
    image_uri = entry['data']['image']
    annotations = entry['annotations']
    bucket_name = image_uri.split('/')[2]
    object_key = '/'.join(image_uri.split('/')[3:])

    images_dest = './Dataset/train/images/'
    download_path = image_uri.split('/')[-1] # Specify the path where you want to save the downloaded image
    if not os.path.exists(images_dest):
        os.makedirs(images_dest)

    print(images_dest+download_path)
    s3_client.download_file(bucket_name, object_key, images_dest+download_path)

    labels_list = np.array([list(annotation['result'][0]['value'].values())[:4] for annotation in annotations])
    labels = np.zeros((labels_list.shape[0], labels_list.shape[1]+1))
    labels[:,1:] = labels_list
    labels_dest = './Dataset/train/labels/'
    if not os.path.exists(labels_dest):
        os.makedirs(labels_dest)

    print(labels_dest+download_path.split('.')[0]+'.txt')
    np.savetxt(labels_dest+download_path.split('.')[0]+'.txt', labels, fmt='%0.5f')
