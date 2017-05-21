import re, glob, random, json, os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_json():
    """loads the json file indicating number of signs
    
    Args:
    None
    
    Returns:
    dictionary loaded from result.json
    """
    with open('result.json') as data_file:    
        data = json.load(data_file)
    return data

def plot_image(img):
    """plots the given image

    Args:
    np array of size 32x32

    Returns:
    None
    """
    plt.imshow(img)
    plt.axis("off")

def get_data():
    """gets all the images and corresponding labels in output folder

    Args:
    None

    Returns:
    data : np array of size (num of images, 32, 32, 3)
    labels : np array of size (num of images)
    """
    images_path = glob.glob("./output/*.png")
    random.shuffle(images_path)
    n = len(images_path)
    data = np.zeros(shape=(n,32,32,3))
    labels = []
    idx = 0
    while images_path:
        #get the image file name without the root folder
        image_path = images_path.pop()
        image_name = image_path.split("/")[2]
        
        #get the sign name (label)
        r = re.compile("([a-zA-Z]+)([0-9]+)")
        m = r.match(image_name)
        label = m.group(1)
    
        image = cv2.imread(image_path)
        
        data[idx,:,:,:] = image
        labels.append(label)
        idx += 1
    labels = np.array(labels)
    return data, labels

def get_class_data(classnames):
    """get data with labels in classnames

    Args:
    a list of classes

    Returns:
    data and labels with those labels
    """
    data, labels = get_data()
    indexes = [i for i,x in enumerate(labels) if x in classnames]
    return data[indexes], labels[indexes]


def main():
    img_dist = load_json()
    max_classes = sorted(img_dist, key=img_dist.get, reverse=True)[:3]
    data, labels = get_class_data(max_classes)

if __name__ == "__main__":
    main()
