import pytest
import os
import shutil

@pytest.fixture(scope="session")
def test_data_dir():
    """Crée un répertoire de données de test"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    test_data = os.path.join(base_dir, "tests", "test_data", "fixtures")
    os.makedirs(test_data, exist_ok=True)
    return test_data 