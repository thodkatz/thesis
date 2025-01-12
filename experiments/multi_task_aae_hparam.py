import json
from thesis import SAVED_RESULTS_PATH
from thesis.multi_task_aae import (
    MultiTaskAae,
    MultiTaskAaeAdversarialAndOptimalTransportTrainer,
    MultiTaskAdversarialTrainer,
    MultiTaskAdversarialGaussianAutoencoderTrainer,
    MultiTaskAdversarialOptimalTransportTrainer,
    MultiTaskVae,
    MultiTaskVaeAdversarialTrainer,
    run_multi_task_adversarial_aae,
)
import optuna

def objective(trial):
    autoencoder_pretrain_epochs = trial.suggest_int(
        "autoencoder_pretrain_epochs", 100, 500, step=100
    )
    #autoencoder_pretrain_epochs = 100

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-4)
    #learning_rate = 1e-4

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    #batch_size = 256

    hidden_layers_autoencoder = trial.suggest_categorical(
        "hidden_layers_autoencoder",
        [
            json.dumps(layer)
            for layer in [
                [32, 16],
                [64, 32],
                [128, 64],
                [256, 128],
                [512, 256],
                [64, 32, 16],
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128],
                [128, 64, 32, 16]
            ]
        ],
    )
    hidden_layers_autoencoder = json.loads(hidden_layers_autoencoder)
    #hidden_layers_autoencoder = [512, 256, 128]
    #hidden_layers_autoencoder = [64, 32]

    num_layers_film = trial.suggest_int("num_layers_film", 0, 1)
    hidden_layers_film = [
        trial.suggest_categorical(f"layer_{i}_size_film", [16, 32, 64])
        for i in range(num_layers_film)
    ]
    #hidden_layers_film = []

    #adversarial_epochs = trial.suggest_int("adversarial_epochs", 0, 1000, step=100) # start from 0 for the non-adversarial case
    adversarial_epochs = 0

    if adversarial_epochs == 0:
        coeff_adversarial = 0
        discriminator_pretrain_epochs = 0
        hidden_layers_discriminator = []
    else:
        coeff_adversarial = trial.suggest_categorical(
            "coeff_adversarial",
            [
                0.005,
                0.01,
                0.03,
                0.05,
                0.07,
                0.09,
            ],
        )
        #coeff_adversarial = 0.01

        discriminator_pretrain_epochs = trial.suggest_int(
            "discriminator_pretrain_epochs", 100, 500, step=100
        )
        #discriminator_pretrain_epochs = 100

        num_layers_discriminator = trial.suggest_int("num_layers_discriminator", 1, 2)
        hidden_layer_discriminator = trial.suggest_categorical(
            "layer_size_discriminator", [16, 32, 64, 128]
        )
        hidden_layers_discriminator = [hidden_layer_discriminator for _ in range(num_layers_discriminator)]
        #hidden_layers_discriminator = [32, 32]

    seed = trial.suggest_categorical("seed", [1, 2, 3, 4])
    #seed = 1
    
    mask_rate = trial.suggest_categorical("mask_rate", [0.1, 0.2, 0.3, 0.4, 0.5])
    #mask_rate = 0.5
    
    dropout_rate = trial.suggest_categorical("dropout_rate", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    #dropout_rate = 0.1

    return run_multi_task_adversarial_aae(
        batch_size=batch_size,
        learning_rate=learning_rate,
        coeff_adversarial=coeff_adversarial,
        autoencoder_pretrain_epochs=autoencoder_pretrain_epochs,
        discriminator_pretrain_epochs=discriminator_pretrain_epochs,
        adversarial_epochs=adversarial_epochs,
        hidden_layers_autoencoder=hidden_layers_autoencoder,
        hidden_layers_discriminator=hidden_layers_discriminator,
        hidden_layers_film=hidden_layers_film,
        seed=seed,
        dropout_rate=dropout_rate,
        mask_rate=mask_rate,
        model_class=MultiTaskVae,
        trainer_class=MultiTaskVaeAdversarialTrainer,
        saved_results_path=SAVED_RESULTS_PATH
    )



if __name__ == "__main__":
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize"],
        study_name="multi_task_vae",
        storage="sqlite:////g/kreshuk/katzalis/repos/thesis-tmp/optuna/db.sqlite3",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1)

    print("Number of finished trials: ", len(study.trials))

    print("Pareto front:")

    trials = sorted(study.best_trials, key=lambda t: t.values)

    for trial in trials:
        print("Trial#{}".format(trial.number))
        print(f"Values: {trial.values}")
        print(f"Params: {trial.params}")
