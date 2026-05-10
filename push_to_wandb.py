import os
import wandb

# Load environment variable for API Key
wandb_api_key = os.environ.get("WANDB_API_KEY")

if wandb_api_key:
    wandb.login(key=wandb_api_key)

# Initialize a new W&B run in the requested project
wandb.init(project="distilbert-goodreads-genres", name="model-upload-run")

# Create a W&B Artifact for the model
artifact = wandb.Artifact(name="distilbert-model", type="model")

# Add the directory where the model was saved locally
model_dir = "distilbert-reviews-genres"
if os.path.exists(model_dir):
    artifact.add_dir(model_dir)
    # Log the artifact to W&B
    wandb.log_artifact(artifact)
    print(f"Model successfully pushed to W&B project: distilbert-goodreads-genres")
else:
    print(f"Error: Model directory '{model_dir}' not found. Ensure training is finished.")

wandb.finish()
