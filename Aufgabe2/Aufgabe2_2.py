"""
    Task 2 of Machine Learning
"""

#===== IMPORTS =======================================
import os
import pandas as pd
from skimage import io, transform

#===== MAIN ==========================================
if __name__ == "__main__":


    # Task 1: Get all pictures of people with more than 70 pics
    pic_data = "lfw_funneled"
    celeb_list = []
    for celeb in os.listdir(pic_data):

        try:
            if len(os.listdir(f"{pic_data}/{celeb}")) >= 70:
                celeb_list.append(celeb)
        except NotADirectoryError:
            continue



    # Read in the pictures
    design_matrix = {
        "label" : [],
        "data" : []
    }
    test_pics = {
        "label" : [],
        "data" : []
    }

    for celeb in celeb_list:
        for index, pic in enumerate(os.listdir(f"{pic_data}/{celeb}")):

                # Read in the picture
                image = io.imread(f"{pic_data}/{celeb}/{pic}", as_gray = True)
                image = transform.resize(image, (32, 32), anti_aliasing=True)
                image = image.flatten()

                # Save the test picture
                if index == 0:
                    test_pics["label"].append(celeb)
                    test_pics["data"].append(image)
                else:
                    design_matrix["label"].append(celeb)
                    design_matrix["data"].append(image)

    design_matrix = pd.DataFrame(design_matrix)
    test = pd.DataFrame(test_pics)
    
    # Perform PCA





    