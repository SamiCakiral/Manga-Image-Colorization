import os
import pytest
import shutil
from src.data.dataset_retriever import DatasetRetriever

@pytest.fixture
def test_dataset_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests"""
    dataset_dir = tmp_path / "test_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    yield dataset_dir
    # Nettoyage après les tests
    shutil.rmtree(dataset_dir)

@pytest.fixture
def test_retriever(test_dataset_dir):
    """Crée une instance de DatasetRetriever pour les tests"""
    return DatasetRetriever(
        gdrive_url="https://test-url.com/test.zip",
        target_images=10,
        dataset_dir=str(test_dataset_dir)
    )

def test_init(test_retriever, test_dataset_dir):
    """Teste l'initialisation du DatasetRetriever"""
    assert test_retriever.target_images == 10
    assert test_retriever.dataset_dir == str(test_dataset_dir)
    assert os.path.exists(test_retriever.paths['bw'])
    assert os.path.exists(test_retriever.paths['color'])
    assert os.path.exists(test_retriever.paths['metadata'])

def test_validate_image(test_retriever):
    """Teste la validation des images"""
    from PIL import Image
    import numpy as np

    # Créer une image valide
    valid_img = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8))
    is_valid, message = test_retriever.validate_image(valid_img)
    assert is_valid
    assert message == "OK"

    # Créer une image trop petite
    small_img = Image.fromarray(np.random.randint(0, 255, (200, 200), dtype=np.uint8))
    is_valid, message = test_retriever.validate_image(small_img)
    assert not is_valid
    assert "trop petite" in message

def test_get_filtered_images(test_retriever, test_dataset_dir):
    """Teste le filtrage des images"""
    # Créer une structure de test
    chapter_dir = os.path.join(test_dataset_dir, "chapter1")
    os.makedirs(chapter_dir)
    
    # Créer quelques images de test
    for i in range(20):
        with open(os.path.join(chapter_dir, f"page_{i:03d}.jpg"), "w") as f:
            f.write("test")

    filtered = test_retriever.get_filtered_images(test_dataset_dir)
    assert len(filtered) > 0
    assert all(isinstance(item, tuple) for item in filtered)
    assert all(len(item) == 2 for item in filtered) 