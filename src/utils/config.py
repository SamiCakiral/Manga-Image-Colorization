from dataclasses import dataclass
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import torch
import re

class ConfigurationError(Exception):
    pass

class Config:
    def __init__(self, config_name: str = "default"):
        """
        Initialise la configuration.
        
        Args:
            config_name: Nom de la configuration à charger (sans extension .yaml)
        """
        self.root_dir = Path(__file__).parent.parent.parent.absolute()
        self.config_dir = self.root_dir / "configs"
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Répertoire de configuration non trouvé: {self.config_dir}")
        
        self.config_name = config_name
        self.config = self._load_config(config_name)
        self._process_config()
        
    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """Charge un fichier de configuration et ses dépendances."""
        # Chercher d'abord dans experiment_configs
        config_path = self.config_dir / "experiment_configs" / f"{config_name}.yaml"
        if not config_path.exists():
            # Si non trouvé, chercher dans le dossier principal
            config_path = self.config_dir / f"{config_name}.yaml"
            if not config_path.exists():
                raise ConfigurationError(f"Configuration '{config_name}' non trouvée")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
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
        # Résoudre les variables
        self.config = self._resolve_variables(self.config)
        
        # Configurer le device
        self.config['training']['device'] = (
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
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

# Instance globale de la configuration
config = Config()
