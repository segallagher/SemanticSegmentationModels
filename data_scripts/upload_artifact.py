import wandb

wandb.login()

api = wandb.Api()
entity = api.default_entity

id = ""

run = wandb.init(
    entity=entity,
    project="autoencoder",
    id=id,
    resume="must",
)

artifact = wandb.Artifact(
    name=f"hyperparameters-{id}",
    type="hyperparameters",
)
artifact.add_file("hyperparameters.json")
run.log_artifact(artifact)
run.finish()
