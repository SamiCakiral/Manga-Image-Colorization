from dataclasses import dataclass
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import re
import platform
import psutil
import subprocess
import sys

class ConfigurationError(Exception):
    pass

class EnvironmentDetector:
    """Détecte l'environnement d'exécution et le matériel disponible."""
    
    @staticmethod
    def is_colab() -> bool:
        """Vérifie si le code s'exécute dans Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Récupère les informations sur le GPU."""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'name': None,
            'memory': None
        }
        
        if gpu_info['available']:
            gpu_info['name'] = torch.cuda.get_device_name(0)
            try:
                gpu_info['memory'] = torch.cuda.get_device_properties(0).total_memory
            except:
                pass
        
        return gpu_info

    @staticmethod
    def get_tpu_info() -> Dict[str, Any]:
        """Vérifie la disponibilité des TPU."""
        tpu_available = False
        tpu_count = 0
        
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            tpu_available = True
            tpu_count = xm.xrt_world_size()
        except ImportError:
            pass
        
        return {
            'available': tpu_available,
            'count': tpu_count
        }

    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """Récupère les informations sur le CPU."""
        return {
            'count': psutil.cpu_count(),
            'physical_count': psutil.cpu_count(logical=False),
            'name': platform.processor(),
            'memory': psutil.virtual_memory().total
        }

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """Récupère toutes les informations sur l'environnement."""
        return {
            'platform': platform.system(),
            'python_version': sys.version.split()[0],
            'torch_version': torch.__version__,
            'is_colab': EnvironmentDetector.is_colab(),
            'gpu': EnvironmentDetector.get_gpu_info(),
            'tpu': EnvironmentDetector.get_tpu_info(),
            'cpu': EnvironmentDetector.get_cpu_info()
        }

class Config:
    def __init__(self, config_name: str = "default"):
        """
        Initialise la configuration.
        
        Args:
            config_name: Nom de la configuration à charger (sans extension .yaml)
        """
        # Détection de l'environnement
        self.env_info = EnvironmentDetector.get_environment_info()
        
        # Configuration des chemins
        self.root_dir = Path(__file__).parent.parent.parent.absolute()
        self.config_dir = self.root_dir / "configs"
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Répertoire de configuration non trouvé: {self.config_dir}")
        
        self.config_name = config_name
        self.config = self._load_config(config_name)
        
        # Ajout des informations d'environnement à la configuration
        self.config['environment'] = self.env_info
        
        # Détermination automatique du device optimal
        self.config['training']['device'] = self._determine_optimal_device()
        
        self._process_config()
        
    def _determine_optimal_device(self) -> torch.device:
        """Détermine le meilleur device disponible pour l'entraînement."""
        if self.env_info['tpu']['available']:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        elif self.env_info['gpu']['available']:
            return torch.device('cuda')
        return torch.device('cpu')

    def print_environment_info(self):
        """Affiche les informations sur l'environnement d'exécution."""
        env = self.env_info
        print("\n🔍 Environnement détecté:")
        print(f"📍 Plateforme: {env['platform']}")
        print(f"🐍 Python: {env['python_version']}")
        print(f"🔥 PyTorch: {env['torch_version']}")
        print(f"☁️  Google Colab: {'Oui' if env['is_colab'] else 'Non'}")
        
        print("\n💻 Ressources disponibles:")
        
        # Info CPU
        cpu = env['cpu']
        print(f"CPU: {cpu['name']}")
        print(f"  - Cœurs: {cpu['physical_count']} physiques, {cpu['count']} logiques")
        print(f"  - RAM: {cpu['memory'] / (1024**3):.1f} GB")
        
        # Info GPU
        gpu = env['gpu']
        if gpu['available']:
            print(f"GPU: {gpu['name']}")
            print(f"  - Nombre: {gpu['count']}")
            if gpu['memory']:
                print(f"  - VRAM: {gpu['memory'] / (1024**3):.1f} GB")
        else:
            print("GPU: Non disponible")
        
        # Info TPU
        tpu = env['tpu']
        if tpu['available']:
            print(f"TPU: Disponible ({tpu['count']} cœurs)")
        else:
            print("TPU: Non disponible")
        
        print(f"\n🎯 Device sélectionné: {self.config['training']['device']}")

    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """Charge un fichier de configuration et ses dépendances."""
        # Pour la configuration par défaut, chercher directement default_config.yaml
        if config_name == "default":
            config_path = self.config_dir / "default_config.yaml"
        else:
            # Chercher d'abord dans experiment_configs
            config_path = self.config_dir / "experiment_configs" / f"{config_name}.yaml"
            if not config_path.exists():
                # Si non trouvé, chercher dans le dossier principal
                config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration '{config_name}' non trouvée dans {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Erreur lors du chargement de la configuration: {str(e)}")
            
        # Gérer l'héritage
        if 'inherit' in config:
            parent_config = self._load_config(config['inherit'])
            config = self._merge_configs(parent_config, config)
            
        return config
    
    def _merge_configs(self, parent: Dict, child: Dict) -> Dict:
        """Fusionne deux configurations en donnant la priorité à l'enfant."""
        merged = parent.copy()
        
        for key, value in child.items():
            if key == 'inherit':
                continue
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def _process_config(self):
        """Traite la configuration après le chargement."""
        # Déterminer le répertoire racine des données selon l'environnement
        if self.env_info['is_colab']:
            data_root = '/content'
        else:
            # Utiliser un répertoire dans le projet pour l'environnement local
            data_root = str(self.root_dir / 'data')
        
        # Ajouter l'information à l'environnement
        self.env_info['data_root'] = data_root
        
        # Résoudre les variables
        self.config = self._resolve_variables(self.config)
        
        # Créer les répertoires nécessaires
        self._create_directories()
    
    def _resolve_variables(self, config: Dict) -> Dict:
        """Résout les variables dans la configuration."""
        def _resolve_value(value: Any) -> Any:
            if isinstance(value, str):
                # Chercher les variables ${var} dans la chaîne
                pattern = r'\${([^}]+)}'
                matches = re.finditer(pattern, value)
                for match in matches:
                    var_name = match.group(1)
                    var_value = self._get_nested_value(var_name)
                    if var_value is not None:
                        value = value.replace(f"${{{var_name}}}", str(var_value))
            elif isinstance(value, dict):
                value = {k: _resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                value = [_resolve_value(v) for v in value]
            return value
        
        return _resolve_value(config)
    
    def _get_nested_value(self, key_path: str) -> Optional[Any]:
        """Récupère une valeur imbriquée dans la configuration."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _create_directories(self):
        """Crée les répertoires nécessaires."""
        paths = self.config.get('paths', {})
        for path in paths.values():
            if isinstance(path, str) and not path.endswith(('.yml', '.yaml', '.txt')):
                os.makedirs(path, exist_ok=True)
    
    def __getattr__(self, name: str) -> Any:
        """Permet d'accéder aux valeurs de configuration comme des attributs."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'{self.__class__.__name__}' n'a pas d'attribut '{name}'")
    
    def save(self, path: Optional[str] = None):
        """Sauvegarde la configuration actuelle."""
        if path is None:
            path = self.config_dir / "experiment_configs" / f"{self.config_name}_saved.yaml"
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

# Instance globale de la configuration avec pattern Singleton
class ConfigSingleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = Config()
        return cls._instance
    
    def __init__(self):
        # S'assurer que print_environment_info() n'est appelé qu'une seule fois
        print("Initialisation de ConfigSingleton")
        if not ConfigSingleton._initialized:
            self.print_environment_info = self._instance.print_environment_info
            self.print_environment_info()
            ConfigSingleton._initialized = True
        
    def __getattr__(self, name):
        return getattr(self._instance, name)

# Créer l'instance unique qui sera utilisée partout
config = ConfigSingleton()
