import argparse
from dataset import diffusion_dataset
from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
import pandas as pd
import torch
import yaml

# Define the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('param_file', type=str, help='Path to the parameters.yml file')
args = parser.parse_args()

# Load parameters from the YAML file
with open(args.param_file, 'r') as file:
    params = yaml.safe_load(file)

output_dir = params['output_dir']
batch_size = params['batch_size']
num_epochs = params['num_epochs']
output_size = params['output_size']
latent_dim = params['latent_dim']
fraction = params['fraction']

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the dataset
diffusion = pd.read_csv("encoding.csv")
#load fraction
diffusion = diffusion.sample(frac=fraction)
train_tensor = diffusion_dataset(diffusion,output_size=output_size)
train_tensor = torch.tensor(train_tensor).to(device)
print(train_tensor.shape)
print(train_tensor[0].shape)

# Define the model
config = BaseTrainerConfig(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_epochs=num_epochs,
)

model_config = AEConfig(
    input_dim=(3, output_size, output_size),
    latent_dim=latent_dim
)

model = AE(
    model_config=model_config,
)

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

pipeline(
    train_data=train_tensor
)