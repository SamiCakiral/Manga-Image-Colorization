from config import config
from AttentionModel import AttentionPointsModel
from ColorizationModel import ColorizationModel
from FusionModule import FusionModule
from TrainingPipeline import TrainingPipeline
from DatasetLoader import DatasetLoader

def main():
    # Initialisation des modèles
    attention_model = AttentionPointsModel()
    primary_model = ColorizationModel(model_type='primary')
    secondary_model = ColorizationModel(model_type='secondary')
    fusion_module = FusionModule()

    # Création du pipeline d'entraînement
    training_pipeline = TrainingPipeline(
        attention_model=attention_model,
        primary_model=primary_model,
        secondary_model=secondary_model
    )

    # Chargement des datasets
    dataset_loader = DatasetLoader(
        bw_dir=config.bw_dir,
        color_dir=config.color_dir,
        metadata_dir=config.metadata_dir
    )

    # Entraînement du modèle d'attention
    training_pipeline.train_attention_model(
        dataloader=dataset_loader,
        epochs=config.num_epochs,
        lr=config.learning_rate
    )

    # Génération des masques d'attention
    training_pipeline.generate_attention_masks(dataset_loader)

    # Extraction et sauvegarde des patches
    patch_extractor = PatchExtractor(patch_size=256, overlap=0)
    training_pipeline.extract_and_save_patches(dataset_loader, patch_extractor, save_dir='path/to/patches')

    # Création des DataLoaders pour les modèles de colorisation
    primary_dataloader = ...  # DataLoader pour les patches importants
    secondary_dataloader = ...  # DataLoader pour les chutes

    # Entraînement des modèles de colorisation
    training_pipeline.train_colorization_models(primary_dataloader, secondary_dataloader, epochs=20, lr=0.0001)

    # Pipeline d'inférence
    inference_pipeline = InferencePipeline(attention_model, primary_model, secondary_model, fusion_module)

    # Colorisation d'une nouvelle image
    bw_image = ...  # Charger une image noir et blanc
    colorized_image = inference_pipeline.colorize_image(bw_image)

if __name__ == "__main__":
    main()