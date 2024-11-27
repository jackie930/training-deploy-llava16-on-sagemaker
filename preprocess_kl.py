## preprocess data
import pandas as pd
#pip install openpyxl
from tqdm import tqdm
import json
import argparse
import os
from PIL import Image
import random

import requests

def download_image(url, filename):
  """Downloads an image from a URL and saves it to a file.

  Args:
    url: The URL of the image.
    filename: The filename to save the image as.
  """
  response = requests.get(url)
  # Check for successful response status code
  if response.status_code == 200:
    with open(filename, 'wb') as f:
      f.write(response.content)
    print(f"Image downloaded successfully: {filename}")
  else:
    print(f"Failed to download image: {url}")


#check if data exists
def check_data(data_root_path, imge_path):
    full_path = os.path.join(data_root_path,imge_path)
    if os.path.isfile(full_path):
        return True
    else:
        return False

#TODO: Add more training data generation methods
def get_single_res(data,idx,output_folder,add_explain=False):
    inputs = data.iloc[idx,:]
    res = {}
    img_name = str(idx)+'.jpg'
    image_path = os.path.join('img',img_name)

    #try to open imae file to see if it exists & valid
    image = Image.open(os.path.join(output_folder,'img',img_name))
    res["id"] = str(idx)
    res['image'] = image_path
    if add_explain:
        res["conversations"] = [{'from': 'human',
                                 'value': '<image>\nYou are an image classifier working on a project to classify images for a hotel website that showcases various amenities and facilities. The goal is to accurately categorize the images to help potential guests better understand the hotel"s offerings. \
                                         After analyzing each image, you should select the appropriate categories from the provided labels: [Hallway,Sauna,Surrounding environment and attractions,Business Center,Pets,Restaurant,Fitness Facility,Activities,'
                                          'Pool,Exterior,Bar,Transportation,Bathhouse,Food and Dining,Lobby,Shopping,Parking,Staircase/Elevator,Laundry Room,Room,Banquet Hall,Spa,Hot springs,Terrace/Patio,Entertainment facility,Other], What"s the classification result? '},
                                {'from': 'gpt',
                                 'value': 'The classification result is [{}].'.format(inputs["label"])}]

    else:
        #https://www.crcv.ucf.edu/wp-content/uploads/2018/11/Misaki-Final-report.pdf
        res["conversations"] = [{'from': 'human',
                                 'value': '<image>\nFill in the blank: this is a photo of a {}, you should select the appropriate categories from the provided labels: [Hallway,Sauna,Surrounding environment and attractions,Business Center,Pets,Restaurant,Fitness Facility,Activities,'
                                          'Pool,Exterior,Bar,Transportation,Bathhouse,Food and Dining,Lobby,Shopping,Parking,Staircase/Elevator,Laundry Room,Room,Banquet Hall,Spa,Hot springs,Terrace/Patio,Entertainment facility,Other]'},
                                {'from': 'gpt',
                                 'value': 'this is a photo of {}.'.format(inputs["label"])}]



    return res

def main(data_path,output_folder):
    data = pd.read_csv(data_path)
    print ("<<< load excel data!")
    data_root_path = os.path.dirname(data_path)

    #make output directy
    os.makedirs(os.path.join(output_folder,'img'),exist_ok=True)

    #download all data
    # print ("<<< download data!")
    '''
    for i in tqdm(range(len(data))):
        # Example usage
        image_url = data['url'][i]
        filename = "{}.jpg".format(i)
        download_image(image_url, os.path.join(output_folder,'img',filename))
    '''

    res_json = []
    # print ("<<< process data!")
    for i in tqdm(range(len(data))):
    #for i in tqdm(range(60)):
        res = get_single_res(data,i,output_folder,add_explain=False)
        if res!={}:
            res_json.append(res)

    #output json, train/test split
    #shuffle data
    random.shuffle(res_json)
    total_len = len(res_json)
    split_idx = int(total_len*0.5)
    train_json = res_json[:split_idx]
    test_json = res_json[split_idx:]

    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join(output_folder, 'data_v2.json'), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(train_json, f)  # Optional parameter for indentation
    print('Data written to json')

    with open(os.path.join(output_folder, 'test_v2.json'), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(test_json, f)  # Optional parameter for indentation
    print('Data written to json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_folder', type=str)
    args = parser.parse_args()
    main(args.data_path, args.output_folder)

