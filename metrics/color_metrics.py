import numpy as np
import cv2
from typing import Dict, Tuple
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from ..src.utils.config import config

class ColorMetrics:
    """
    Classe pour calculer les métriques spécifiques à la colorisation.
    Implémente les trois métriques principales :
    - Précision des couleurs
    - Cohérence des transitions
    - Qualité globale
    """
    
    def __init__(self):
        """Initialise les métriques de base"""
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        
    def compute_color_accuracy(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, Dict]:
        """
        Calcule la précision des couleurs entre l'image prédite et la cible.
        Utilise l'espace colorimétrique LAB pour une meilleure perception des différences.
        
        Arguments:
            pred (np.ndarray): Image prédite (RGB, [0,1])
            target (np.ndarray): Image cible (RGB, [0,1])
            
        Returns:
            score (float): Score global de précision des couleurs [0,1]
            details (Dict): Détails des métriques par canal et région
        """
        # Conversion en LAB
        pred_lab = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Calcul des différences par canal
        l_diff = np.mean(np.abs(pred_lab[:,:,0] - target_lab[:,:,0])) / 100.0
        a_diff = np.mean(np.abs(pred_lab[:,:,1] - target_lab[:,:,1])) / 127.0
        b_diff = np.mean(np.abs(pred_lab[:,:,2] - target_lab[:,:,2])) / 127.0
        
        # Score global normalisé
        score = 1.0 - (0.3 * l_diff + 0.35 * a_diff + 0.35 * b_diff)
        
        # Analyse par région (zones claires, moyennes, sombres)
        luminance = target_lab[:,:,0] / 100.0
        regions = {
            'dark': luminance < 0.3,
            'mid': (luminance >= 0.3) & (luminance < 0.7),
            'bright': luminance >= 0.7
        }
        
        region_scores = {}
        for region_name, mask in regions.items():
            if np.any(mask):
                region_scores[region_name] = {
                    'l_diff': np.mean(np.abs(pred_lab[:,:,0][mask] - target_lab[:,:,0][mask])) / 100.0,
                    'a_diff': np.mean(np.abs(pred_lab[:,:,1][mask] - target_lab[:,:,1][mask])) / 127.0,
                    'b_diff': np.mean(np.abs(pred_lab[:,:,2][mask] - target_lab[:,:,2][mask])) / 127.0
                }
        
        details = {
            'global_differences': {
                'l_diff': l_diff,
                'a_diff': a_diff,
                'b_diff': b_diff
            },
            'region_scores': region_scores
        }
        
        return score, details

    def compute_transition_coherence(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, Dict]:
        """
        Évalue la cohérence des transitions de couleur.
        Analyse les gradients de couleur et leur continuité.
        
        Arguments:
            pred (np.ndarray): Image prédite (RGB, [0,1])
            target (np.ndarray): Image cible (RGB, [0,1])
            
        Returns:
            score (float): Score de cohérence des transitions [0,1]
            details (Dict): Détails des métriques de transition
        """
        def compute_gradients(img):
            # Conversion en LAB
            lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            
            # Calcul des gradients pour chaque canal
            gradients = []
            for i in range(3):
                gx = cv2.Sobel(lab[:,:,i], cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(lab[:,:,i], cv2.CV_32F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(gx**2 + gy**2)
                gradient_direction = np.arctan2(gy, gx)
                gradients.append((gradient_magnitude, gradient_direction))
            return gradients
        
        # Calcul des gradients
        pred_gradients = compute_gradients(pred)
        target_gradients = compute_gradients(target)
        
        # Comparaison des magnitudes et directions
        magnitude_scores = []
        direction_scores = []
        
        for i in range(3):
            pred_mag, pred_dir = pred_gradients[i]
            target_mag, target_dir = target_gradients[i]
            
            # Score de magnitude
            magnitude_diff = np.abs(pred_mag - target_mag) / (target_mag + 1e-6)
            magnitude_scores.append(np.mean(1 - np.clip(magnitude_diff, 0, 1)))
            
            # Score de direction (en tenant compte de la circularité)
            direction_diff = np.minimum(
                np.abs(pred_dir - target_dir),
                2 * np.pi - np.abs(pred_dir - target_dir)
            ) / np.pi
            direction_scores.append(np.mean(1 - direction_diff))
        
        # Calcul du score global
        weights = [0.3, 0.35, 0.35]  # Poids pour L*, a*, b*
        magnitude_score = np.average(magnitude_scores, weights=weights)
        direction_score = np.average(direction_scores, weights=weights)
        
        score = 0.6 * magnitude_score + 0.4 * direction_score
        
        details = {
            'magnitude_scores': dict(zip(['l', 'a', 'b'], magnitude_scores)),
            'direction_scores': dict(zip(['l', 'a', 'b'], direction_scores)),
            'magnitude_score': magnitude_score,
            'direction_score': direction_score
        }
        
        return score, details

    def compute_global_quality(self, pred: np.ndarray, target: np.ndarray) -> Tuple[float, Dict]:
        """
        Calcule un score global de qualité combinant plusieurs métriques.
        
        Arguments:
            pred (np.ndarray): Image prédite (RGB, [0,1])
            target (np.ndarray): Image cible (RGB, [0,1])
            
        Returns:
            score (float): Score global de qualité [0,1]
            details (Dict): Détails des différentes composantes
        """
        # Calcul des métriques individuelles
        color_score, color_details = self.compute_color_accuracy(pred, target)
        transition_score, transition_details = self.compute_transition_coherence(pred, target)
        
        # Conversion en tenseurs pour PSNR et SSIM
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
        target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
        
        # Calcul PSNR et SSIM
        psnr_score = self.psnr(pred_t, target_t).item() / 100.0  # Normalisation
        ssim_score = self.ssim(pred_t, target_t).item()
        
        # Calcul du score global avec pondération
        weights = {
            'color_accuracy': 0.35,
            'transition_coherence': 0.35,
            'structural_similarity': 0.20,
            'signal_quality': 0.10
        }
        
        score = (
            weights['color_accuracy'] * color_score +
            weights['transition_coherence'] * transition_score +
            weights['structural_similarity'] * ssim_score +
            weights['signal_quality'] * psnr_score
        )
        
        details = {
            'component_scores': {
                'color_accuracy': color_score,
                'transition_coherence': transition_score,
                'ssim': ssim_score,
                'psnr': psnr_score
            },
            'weights': weights,
            'color_details': color_details,
            'transition_details': transition_details
        }
        
        return score, details

    def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray) -> Dict:
        """
        Calcule toutes les métriques disponibles.
        
        Arguments:
            pred (np.ndarray): Image prédite (RGB, [0,1])
            target (np.ndarray): Image cible (RGB, [0,1])
            
        Returns:
            metrics (Dict): Dictionnaire contenant toutes les métriques et leurs détails
        """
        color_score, color_details = self.compute_color_accuracy(pred, target)
        transition_score, transition_details = self.compute_transition_coherence(pred, target)
        global_score, global_details = self.compute_global_quality(pred, target)
        
        return {
            'scores': {
                'color_accuracy': color_score,
                'transition_coherence': transition_score,
                'global_quality': global_score
            },
            'details': {
                'color': color_details,
                'transition': transition_details,
                'global': global_details
            }
        } 