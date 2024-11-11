import xml.etree.ElementTree as ET
from PIL import Image
import os
from time import sleep
from tqdm import tqdm
import pandas as pd

# Paths
xml_dir = 'C:/Users/tirth/Downloads/Compressed/Manga109/Manga109_released_2023_12_07/annotations/'  # Directory containing XML files
image_dir = 'C:/Users/tirth/Downloads/Compressed/Manga109/Manga109_released_2023_12_07/images/'  # Directory containing page images

# Helper function to convert bounding box to x1, y1, x2, y2, x3, y3, x4, y4
def bbox_to_coordinates(xmin, ymin, xmax, ymax):
    return [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]

images=[]
annotations = []

# Iterate over all XML files
for xml_file in tqdm(os.listdir(xml_dir)):
    if not xml_file.endswith('.xml'):
        continue
    # print(xml_file)
    # sleep(5)
    # Parse the XML file
    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()
    
    # Process each page in the XML
    for page in root.find('pages'):
        page_index = page.get('index')
        
        # Load the corresponding image for the page
        page_image_path = os.path.join(image_dir, f'{xml_file.split('.')[0]}/{str(page_index).zfill(3)}.jpg')
        images.append(page_image_path)
        if not os.path.exists(page_image_path):
            print(f"Warning: Image for page {page_index} not found.")
            continue
        
        # Open the page image
        # page_image = Image.open(page_image_path)
        
        # Prepare data for the first model (page-level data with annotations)
        annots=[]
        for text in page.findall('text'):
            # Extract text annotation coordinates
            xmin, ymin = int(text.get('xmin')), int(text.get('ymin'))
            xmax, ymax = int(text.get('xmax')), int(text.get('ymax'))
            coordinates = bbox_to_coordinates(xmin, ymin, xmax, ymax)
            annots.append(coordinates)
        annotations.append(annots)
    
pd.DataFrame({'images':images,'annotations':annotations}).to_csv(os.path.join(os.getcwd(),'model_east_metadata.csv'))
            
