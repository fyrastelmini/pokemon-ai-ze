from dataset import diffusion_dataset
from pythae.models import WAE_MMD, WAE_MMD_Config
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
import pandas as pd

output_size=512
# Load the dataset
diffusion = pd.read_csv("encoding.csv")
train_tensor = diffusion_dataset(diffusion,output_size=output_size)

# Define the model
config = BaseTrainerConfig(
    output_dir='my_model',
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=10, # Change this to train the model a bit more
)


model_config = WAE_MMD_Config(
    input_dim=(3, output_size, output_size),
    latent_dim=128,
    kernel_choice='imq',
    reg_weight=100,
    kernel_bandwidth=2
)

model = WAE_MMD(
    model_config=model_config,
)

pipeline = TrainingPipeline(
    training_config=config,
    model=model
)

pipeline(
    train_data=train_tensor
)