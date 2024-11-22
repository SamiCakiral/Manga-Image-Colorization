import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import config

def diversity_loss(points):
    """
    Encourage une distribution spatiale plus diversifiée des points.
    
    Args:
        points (torch.Tensor): Points d'attention [B, N, 2]
    """
    # Calculer les distances entre tous les points
    dists = torch.cdist(points, points)
    
    # Masquer la diagonale
    mask = ~torch.eye(points.size(1), dtype=torch.bool, device=points.device)
    dists = dists.masked_fill(~mask.unsqueeze(0), float('inf'))
    
    # Pénaliser les points trop proches
    min_dists = dists.min(dim=-1)[0]
    return torch.mean(torch.exp(-min_dists / 0.1))

def compute_grid_alignment_loss(points, grid_size):
    """
    Encourage l'alignement des points avec une grille détectée.
    
    Args:
        points (torch.Tensor): Points d'attention [B, N, 2]
        grid_size (torch.Tensor): Taille de la grille détectée [B]
    """
    batch_size = points.size(0)
    device = points.device
    
    # Créer une grille régulière basée sur la taille détectée
    grid_loss = torch.tensor(0., device=device)
    
    for b in range(batch_size):
        # Créer la grille pour cette image
        step = 1.0 / grid_size[b].item()
        grid_x = torch.arange(0, 1.0, step=step, device=device)
        grid_y = torch.arange(0, 1.0, step=step, device=device)
        
        grid_points_x, grid_points_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_points = torch.stack([grid_points_x.flatten(), grid_points_y.flatten()], dim=1)
        
        # Calculer la distance minimale de chaque point à la grille
        dists = torch.cdist(points[b], grid_points)
        min_dists = dists.min(dim=1)[0]
        
        grid_loss += torch.mean(min_dists)
    
    return grid_loss / batch_size

def compute_coverage_loss(points):
    """
    Encourage une meilleure couverture de l'image.
    
    Args:
        points (torch.Tensor): Points d'attention [B, N, 2]
    """
    # Distance au centre (0.5, 0.5)
    center_dist = torch.norm(points - 0.5, dim=-1)
    
    # Pénaliser les points trop proches du centre ou des bords
    coverage_loss = torch.mean(
        torch.exp(-center_dist) +  # Éviter la concentration au centre
        torch.exp(-(1 - center_dist))  # Éviter les bords
    )
    
    return coverage_loss

class AttentionLoss(nn.Module):
    """
    Classe de perte améliorée pour le modèle d'attention.
    """
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.weights = config.config['model']['attention']['loss_weights']
    
    def forward(self, points, scores, grid_present, grid_size, complexity_map=None):
        """
        Calcule la perte totale avec tous les termes d'amélioration.
        
        Args:
            points (torch.Tensor): Points d'attention [B, N, 2]
            scores (torch.Tensor): Scores d'attention [B, N]
            grid_present (torch.Tensor): Indicateur de présence de grille [B]
            grid_size (torch.Tensor): Taille de la grille détectée [B]
            complexity_map (torch.Tensor, optional): Carte de complexité [B, H, W]
        """
        # Perte de diversité spatiale
        diversity_loss = diversity_loss(points)
        
        # Perte de couverture
        coverage_loss = compute_coverage_loss(points)
        
        # Perte d'alignement avec la grille (seulement si une grille est détectée)
        grid_loss = torch.tensor(0., device=points.device)
        if grid_present.any():
            grid_loss = compute_grid_alignment_loss(
                points[grid_present],
                grid_size[grid_present]
            )
        
        # Perte de concentration des scores
        score_concentration_loss = -torch.mean(
            scores * torch.log(scores + 1e-8) + 
            (1 - scores) * torch.log(1 - scores + 1e-8)
        )
        
        # Perte de complexité si une carte est fournie
        complexity_loss = torch.tensor(0., device=points.device)
        if complexity_map is not None:
            # Échantillonner la carte de complexité aux positions des points
            B, H, W = complexity_map.shape
            points_scaled = points.clone()
            points_scaled[:, :, 0] *= W
            points_scaled[:, :, 1] *= H
            points_scaled = points_scaled.long()
            
            sampled_complexity = complexity_map[
                torch.arange(B).unsqueeze(1),
                points_scaled[:, :, 1],
                points_scaled[:, :, 0]
            ]
            
            complexity_loss = -torch.mean(sampled_complexity * scores)
        
        # Perte totale avec pondération
        total_loss = (
            self.weights['diversity'] * diversity_loss +
            self.weights['coverage'] * coverage_loss +
            self.weights['grid'] * grid_loss +
            self.weights['score_concentration'] * score_concentration_loss +
            self.weights.get('complexity', 0.5) * complexity_loss
        )
        
        return total_loss, {
            'diversity': diversity_loss.item(),
            'coverage': coverage_loss.item(),
            'grid': grid_loss.item(),
            'score_concentration': score_concentration_loss.item(),
            'complexity': complexity_loss.item()
        }

# Créer une instance globale de la perte améliorée
attention_criterion = AttentionLoss()