import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import config

def diversity_loss(points, scores):
    """Pénalise les points trop proches les uns des autres"""
    batch_size = points.size(0)
    distances = torch.cdist(points, points)
    
    # Masquer la diagonale
    mask = torch.eye(points.size(1), device=points.device).bool()
    distances = distances.masked_fill(mask.unsqueeze(0), float('inf'))
    
    # Pénaliser les points trop proches
    min_distances = distances.min(dim=-1)[0]
    diversity_loss = torch.exp(-min_distances).mean()
    
    return diversity_loss

def attention_loss(points, scores, features_map):
    """Combine plusieurs termes de perte pour un apprentissage efficace"""
    # Perte de diversité spatiale
    div_loss = diversity_loss(points, scores)
    
    # Perte de couverture de l'image
    coverage_loss = torch.mean((points - 0.5).pow(2).sum(dim=-1))
    
    # Perte de pertinence (à implémenter selon vos besoins)
    # ...
    
    return div_loss + 0.1 * coverage_loss

class AttentionLoss(nn.Module):
    """Classe de perte pour le modèle d'attention"""
    
    def __init__(self):
        super(AttentionLoss, self).__init__()
        
    def forward(self, points, scores, target=None):
        """
        Calcule la perte totale pour le modèle d'attention.
        
        Args:
            points (torch.Tensor): Points d'attention prédits (B, N, 2)
            scores (torch.Tensor): Scores d'attention prédits (B, N)
            target (torch.Tensor, optional): Features map cible si disponible
            
        Returns:
            torch.Tensor: Perte totale
        """
        # Perte de diversité
        div_loss = diversity_loss(points, scores)
        
        # Perte de couverture
        coverage_loss = torch.mean((points - 0.5).pow(2).sum(dim=-1))
        
        # Perte de concentration des scores
        score_concentration_loss = -torch.mean(
            scores * torch.log(scores + 1e-8) + 
            (1 - scores) * torch.log(1 - scores + 1e-8)
        )
        
        # Perte totale avec pondération
        total_loss = (
            config.config['model']['attention']['loss_weights']['diversity'] * div_loss +
            config.config['model']['attention']['loss_weights']['coverage'] * coverage_loss +
            config.config['model']['attention']['loss_weights']['score_concentration'] * score_concentration_loss
        )
        
        return total_loss

# Créer une instance globale de la perte
attention_criterion = AttentionLoss() 