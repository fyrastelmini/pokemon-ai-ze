import pandas as pd
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import torch
from tqdm import tqdm

def generate_dataset(sprites_path="sprites/sprites/pokemon/", save=False):
    def load_sprites():
        # get the path of all .png files under sprites/
        paths = []
        for root, dirs, files in os.walk(sprites_path):
            for file in files:
                if file.endswith(".png"):
                    paths.append(os.path.join(root, file))
        # get the size of each image
        sizes = []
        pbar = tqdm(paths)
        for path in paths:
            try:
                img = plt.imread(path)
                sizes.append(img.shape[1::-1])
                pbar.set_description(f"loading {path}")
                pbar.update(1)
            except:
                #remove the path from the list if the image is not readable
                sizes.append(np.nan)
                pbar.set_description(f"removing {path}")
                pbar.update(1)
        pbar.close()
        print(len(paths), len(sizes))
        # create a dataframe
        df = pd.DataFrame({"path": paths, "size": sizes})
        return df

    df = load_sprites()
    df = df.dropna()
    df["ratio"]= df["size"].apply(lambda x: x[0]/x[1])
    df= df[df["ratio"]==1.0]
    df["dim"]= df["size"].apply(lambda x: x[0])
    df["train"] = df["dim"]<200
    # view the files with similar filenames
    df["filename"] = df["path"].apply(lambda x: x.split("/")[-1])
    df["filename"].value_counts()
    diffusion = df[df["train"]==True]
    encoding = df[df["train"]==False]
    # drop the columns with unique filenames, keep the duplicated, for the train colums only
    diffusion = diffusion[diffusion["filename"].duplicated(keep=False)]
    # save the datasets
    if save:
        diffusion.to_csv("diffusion.csv", index=False)
        encoding.to_csv("encoding.csv", index=False)
    return diffusion, encoding

def diffusion_dataloader(dataframe, batch_size=32, output_size=128):
    # from a dataset of paths to .png files, load the images and resize them to output_size, format them as a torch tensor with batch size
    for index in range(0, len(dataframe), batch_size):
        batch = dataframe.iloc[index:index+batch_size]
        images = []
        for path in batch["path"]:
            img = Image.open(path)
            img = img.resize((output_size, output_size))  # Resize the image
            img = np.array(img)[:,:,:3]  # Convert the image to numpy array and keep only the first 3 channels
            img = img/255
            img = torch.tensor(img).permute(2,0,1)

            images.append(img)
        images = torch.stack(images)
        yield images

def encoding_dataloader(dataframe, batch_size=32, output_size=512):
    # from a dataset of paths to .png files, load the images and resize them to output_size, format them as a tensor with batch size
    for index in range(0, len(dataframe), batch_size):
        batch = dataframe.iloc[index:index+batch_size]
        images = []
        for path in batch["path"]:
            img = Image.open(path)
            img = img.resize((output_size, output_size))  # Resize the image
            img = np.array(img)[:,:,:3]  # Convert the image to numpy array and keep only the first 3 channels
            img = img/255
            img = torch.tensor(img).permute(2,0,1)

            images.append(img)
        images = torch.stack(images)
        yield images

def diffusion_dataset(dataframe, output_size=8):
    images = []
    pbar= tqdm(dataframe["path"])
    for path in dataframe["path"]:
        img = Image.open(path)
        img = img.resize((output_size, output_size))  # Resize the image
        img = np.array(img)[:,:,:3]  # Convert the image to numpy array and keep only the first 3 channels
        img = img/255

        images.append(img)
        pbar.update(1)
    images = np.stack(images)  # Use numpy's stack function
    pbar.close()
    return images