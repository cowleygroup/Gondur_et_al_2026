import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from PIL import Image

image_directory = "./V4_neural_data/images/"

data_v4_session190923 = np.load('./V4_neural_data/responses_repeat_averaged/responses_190923.npy').T
data_v4_session201025 = np.load('./V4_neural_data/responses_repeat_averaged/responses_201025.npy').T
data_v4_session211022 = np.load('./V4_neural_data/responses_repeat_averaged/responses_211022.npy').T
data_v4_session210225 = np.load('./V4_neural_data/responses_repeat_averaged/responses_210225.npy').T
data_v4_session210224 = np.load('./V4_neural_data/responses_repeat_averaged/responses_210224.npy').T


def get_images_from_zip(directory_path):
    imgs = []
    counts = 0

    archive = directory_path
    names = archive.namelist()
    sorted(names)
    for name in names:    
        if name.endswith('.jpg'):
            image_data = Image.open(archive.open(name)).resize(size=(224,224))
            img = np.array(image_data) 
            imgs.append(img)
        
            counts += 1    

    archive.close()
    imgs = np.array(imgs)
    return (imgs)

def sort_images_responses(response_dir ):
    sorted_indices = np.argsort(response_dir)
    return sorted_indices

image_zipfile_190923 = zipfile.ZipFile(f'{image_directory}images_190923.zip' )
image_zipfile_201025 = zipfile.ZipFile(f'{image_directory}images_201025.zip' )
image_zipfile_211022 = zipfile.ZipFile(f'{image_directory}images_211022.zip' )
image_zipfile_210225 = zipfile.ZipFile(f'{image_directory}images_210225.zip' )
image_zipfile_210224 = zipfile.ZipFile(f'{image_directory}images_210224.zip' )


all_images_190923 = get_images_from_zip(image_zipfile_190923)
all_images_201025 = get_images_from_zip(image_zipfile_201025)
all_images_211022 = get_images_from_zip(image_zipfile_211022)
all_images_210225 = get_images_from_zip(image_zipfile_210225)
all_images_210224 = get_images_from_zip(image_zipfile_210224)


plt.hist(data_v4_session190923[:,11]/0.1, bins=30)

def plot_images(images, feature_values, n=25):


    for current_neuron in range(89):
        sorted_indices = np.argsort(feature_values[:, current_neuron])
        lowest_indices = sorted_indices[:n]
        highest_indices = sorted_indices[-n:]

        fig, axes = plt.subplots(5, 10, figsize=(20, 10))  

        for i, idx in enumerate(lowest_indices):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            ax.imshow(images[idx])
            ax.axis('off')
        for i, idx in enumerate(highest_indices):
            row = i // 5
            col = i % 5 + 5 
            ax = axes[row, col]
            ax.imshow(images[idx])
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'session_201025_{current_neuron}_pref_antipref_side_by_side.pdf')
        plt.close()


plot_images(images=all_images_201025, feature_values=data_v4_session201025, n=25)
