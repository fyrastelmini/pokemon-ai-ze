import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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