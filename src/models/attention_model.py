import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import functional as TF
from ..utils.config import config
from .losses import AttentionLoss, diversity_loss
import numpy as np

class AttentionPointsModel(nn.Module):
    """Cette classe est un modèle d'attention qui prédit des points clés pour guider le découpage intelligent. 
    Elle est basée sur le modèle ViT-B/16 de PyTorch.
    Et elle utilise une tête d'attention pour prédire les points d'attention Avec une gaussienne autour du point.
    ViT-B/16 est un modèle de vision transformer pré-entraîné sur ImageNet qui est utilisé pour extraire des features.››
    """
    def __init__(self, max_points=32):
        """
        Initialise le modèle d'attention qui prédit des points clés.
        
        Arguments:
        - max_points: Nombre maximum de points d'attention à prédire
        """
        super(AttentionPointsModel, self).__init__()
        
        # Vision Transformer modifié pour prédire des points
        self.vit = models.vit_b_16(
            weights='IMAGENET1K_V1' if config.config['model']['vit']['pretrained'] else None
        )
        
        # Calculer la dimension d'entrée pour spatial_features
        vit_features = self.vit.hidden_dim  # 768 pour ViT-B/16
        edge_features = (224 // 16) * (224 // 16)  # Features des bords après conv
        total_features = vit_features + edge_features
        
        # Ajout de couches intermédiaires pour mieux capturer les caractéristiques spatiales
        self.spatial_features = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Tête d'attention avec sortie plus diversifiée
        self.attention_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, max_points * 3)
        )
        
        self.max_points = max_points
        
        # Initialisation des poids pour éviter la convergence vers le centre
        self._init_weights()
        
    def _init_weights(self):
        """Initialise les poids pour favoriser une distribution spatiale plus diverse"""
        for m in self.spatial_features.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Initialisation spéciale pour la dernière couche
        last_layer = self.attention_head[-1]
        nn.init.uniform_(last_layer.weight, -0.1, 0.1)
        if last_layer.bias is not None:
            # Initialiser les biais pour avoir des points répartis sur toute l'image
            points_bias = torch.linspace(-2, 2, self.max_points * 2)
            scores_bias = torch.zeros(self.max_points)
            nn.init.constant_(last_layer.bias[:self.max_points * 2], 0)
            nn.init.constant_(last_layer.bias[self.max_points * 2:], 0)
    
    def forward(self, x):
        """
        Prédit les points d'attention pour l'image.
        
        Returns:
        - points: tensor de forme (batch_size, max_points, 2) pour les coordonnées x,y
        - scores: tensor de forme (batch_size, max_points) pour les scores d'importance
        """
        # Détecter et recadrer les marges blanches
        x, crop_coords = self.remove_white_margins(x)
        
        # Redimensionner l'image à la taille attendue par le ViT (224x224)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Adapter l'image en entrée
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Détecter les bords des cases avec Canny
        edges = self.detect_panel_edges(x)
        
        # Redimensionner les edges pour correspondre à la taille des features ViT
        edges = F.adaptive_avg_pool2d(edges, (14, 14))  # 224/16 = 14
        edges = edges.view(edges.size(0), -1)  # Aplatir
        
        # Obtenir les features du ViT
        x = self.vit.conv_proj(x)
        batch_size, embed_dim, _, _ = x.shape
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        # Ajouter le token [CLS]
        cls_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Ajouter l'embedding de position
        x = x + self.vit.encoder.pos_embedding
        
        # Passer dans l'encodeur
        for block in self.vit.encoder.layers:
            x = block(x)
        
        # Prendre le token [CLS]
        features = x[:, 0]
        
        # Concaténer avec les features des bords
        features = torch.cat([features, edges], dim=1)
        
        # Extraction des caractéristiques spatiales
        features = self.spatial_features(features)
        
        # Prédiction avec bruit pour plus de diversité
        if self.training:
            features = features + torch.randn_like(features) * 0.1
            
        # Prédire les points et scores
        output = self.attention_head(features)
        output = output.view(-1, self.max_points, 3)
        
        # Séparer coordonnées et scores
        points = output[:, :, :2]
        scores = output[:, :, 2]
        
        # Forcer une distribution plus uniforme des points
        points = self.distribute_points(points)
        
        # Normaliser les scores
        scores = torch.sigmoid(scores)
        
        # Filtrer les points selon leur score
        points, scores = self.filter_attention_points(
            points, 
            scores,
            min_points=5,  # Au moins 5 points
            min_score_threshold=0.3,  # Score minimum de 0.3
            max_points=15  # Maximum 15 points
        )
        
        # Ajuster les coordonnées des points en fonction du recadrage
        points = self.adjust_points_to_original(points, crop_coords)
        
        return points, scores
        
    def get_attention_patches(self, images, points, scores, patch_size=32):
        """
        Extrait les patches autour des points d'attention.
        
        Args:
            images (torch.Tensor): Images d'entrée [B, C, H, W]
            points (torch.Tensor): Points d'attention [B, N, 2]
            scores (torch.Tensor): Scores d'importance [B, N]
            patch_size (int): Taille des patches à extraire
            
        Returns:
            patches (list): Liste de patches pour chaque image
            coords (list): Liste des coordonnées des patches
        """
        B = images.size(0)
        H, W = images.shape[-2:]
        N = points.size(1)  # Utiliser le nombre réel de points
        
        # Convertir les coordonnées normalisées en pixels
        points = points.clone()
        points[:, :, 0] *= W
        points[:, :, 1] *= H
        
        # Trier les points par score d'importance
        scores = scores[:, :N]  # S'assurer que scores a la même taille que points
        
        patches = []
        coords = []
        
        for b in range(B):
            # Trier les points par score
            indices = torch.argsort(scores[b], descending=True)
            sorted_points = points[b][indices]
            
            image_patches = []
            patch_coords = []
            
            for point in sorted_points:
                x, y = point
                
                # Calculer les coordonnées du patch
                x1 = max(0, int(x - patch_size // 2))
                x2 = min(W, int(x + patch_size // 2))
                y1 = max(0, int(y - patch_size // 2))
                y2 = min(H, int(y + patch_size // 2))
                
                # Extraire le patch
                patch = images[b, :, y1:y2, x1:x2]
                
                # Redimensionner si nécessaire
                if patch.size(-1) != patch_size or patch.size(-2) != patch_size:
                    patch = F.interpolate(
                        patch.unsqueeze(0),
                        size=(patch_size, patch_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                image_patches.append(patch)
                patch_coords.append((x1, y1, x2, y2))
            
            patches.append(image_patches)
            coords.append(patch_coords)
        
        return patches, coords
    
    def compute_complexity_maps(self, image):
        """
        Calcule une carte de chaleur de complexité pour guider l'attention.
        """
        # Complexité structurelle (gradients)
        structural = self.compute_structural_complexity(image)
        
        # Complexité des transitions
        transitions = self.compute_transition_complexity(image)
        
        # Combiner les métriques
        complexity = 0.7 * structural + 0.3 * transitions
        
        # Normaliser
        complexity = complexity / complexity.max()
        
        return complexity
    
    @staticmethod
    def compute_structural_complexity(image):
        """Calcule la complexité structurelle de l'image en utilisant les gradients de Sobel.
        
        Arguments:
            image (torch.Tensor): Image d'entrée (B, C, H, W)
            
        Retourne:
            torch.Tensor: Carte de complexité structurelle (B, 1, H, W)
        """
        # Conversion en niveaux de gris avec coefficients ITU-R BT.601
        grayscale = 0.299 * image[:,0,:,:] + 0.587 * image[:,1,:,:] + 0.114 * image[:,2,:,:]
        grayscale = grayscale.unsqueeze(1)
        
        # Noyaux de Sobel pré-calculés
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=image.dtype)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=image.dtype)
        
        # Application des filtres de Sobel
        grad_x = F.conv2d(grayscale, weight=sobel_x.to(image.device), padding=1)
        grad_y = F.conv2d(grayscale, weight=sobel_y.to(image.device), padding=1)
        
        # Magnitude du gradient avec normalisation
        gradients = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
        gradients = gradients / gradients.max()
        
        return gradients
    
    @staticmethod
    def compute_transition_complexity(image):
        """Calcule la complexité des transitions de luminosité dans l'image.
        
        Arguments:
            image (torch.Tensor): Image d'entrée (B, C, H, W)
            
        Retourne:
            torch.Tensor: Carte de complexité des transitions (B, 1, H, W)
        """
        # Conversion en luminance avec coefficients ITU-R BT.601
        luminance = 0.299 * image[:,0,:,:] + 0.587 * image[:,1,:,:] + 0.114 * image[:,2,:,:]
        
        # Calcul des différences horizontales et verticales
        delta_h = torch.abs(luminance[:, :, 1:] - luminance[:, :, :-1])
        delta_v = torch.abs(luminance[:, 1:, :] - luminance[:, :-1, :])
        
        # Padding pour restaurer les dimensions
        delta_h = F.pad(delta_h, (0, 1, 0, 0))
        delta_v = F.pad(delta_v, (0, 0, 0, 1))
        
        # Combinaison des transitions
        transitions = torch.maximum(delta_h, delta_v).unsqueeze(1)
        transitions = transitions / transitions.max()
        
        return transitions
    
    @staticmethod
    def compute_semantic_complexity(image):
        """Calcule la complexité sémantique de l'image.
        
        Arguments:
            image (torch.Tensor): Image d'entrée (B, C, H, W)
            
        Retourne:
            torch.Tensor: Carte de complexité sémantique (B, 1, H, W)
        """
        # Utilisation de DeepLabV3 pré-entraîné pour la segmentation
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        model.eval()
        model = model.to(image.device)
        
        # Normalisation de l'image pour le modèle
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
        normalized_image = (image - mean) / std
        
        with torch.no_grad():
            # Obtenir les prédictions de segmentation
            output = model(normalized_image)['out']
            
            # Calculer la diversité des classes par pixel
            probs = torch.softmax(output, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1, keepdim=True)
            
            # Normaliser l'entropie
            semantic_complexity = entropy / torch.log(torch.tensor(output.shape[1]))
            semantic_complexity = semantic_complexity / semantic_complexity.max()
        
        return semantic_complexity
        
    
    def train_step(self, batch, optimizer, criterion):
        """
        Une étape d'entraînement qui utilise toutes les pertes définies.
        
        Args:
            batch: Dictionnaire contenant les données d'entrée
            optimizer: Optimiseur PyTorch
            criterion: Instance de AttentionLoss
        """
        optimizer.zero_grad()
        
        # Obtenir les images du batch
        images = batch['image']
        
        # Calculer la carte de complexité comme pseudo vérité terrain
        complexity_maps = self.compute_complexity_maps(images)
        
        # Prédire les points d'attention
        points, scores = self(images)
        
        # Convertir les points en carte d'attention
        pred_attention = self.points_to_attention_map(points, scores, images.shape[2:])
        
        # Calculer les différentes composantes de la perte
        # 1. Perte principale (définie dans AttentionLoss)
        main_loss = criterion(points, scores, complexity_maps)
        
        # 2. Perte de diversité spatiale (importée de losses.py)
        div_loss = diversity_loss(points, scores)
        
        # 3. Perte de couverture de l'image
        coverage_loss = torch.mean((points - 0.5).pow(2).sum(dim=-1))
        
        # 4. Perte de concentration des scores
        score_concentration_loss = -torch.mean(
            scores * torch.log(scores + 1e-8) + 
            (1 - scores) * torch.log(1 - scores + 1e-8)
        )
        
        # Perte totale avec pondération depuis la config
        total_loss = (
            main_loss +
            config.config['model']['attention']['loss_weights']['diversity'] * div_loss +
            config.config['model']['attention']['loss_weights']['coverage'] * coverage_loss +
            config.config['model']['attention']['loss_weights']['score_concentration'] * score_concentration_loss
        )
        
        # Ajouter une perte de complexité pour guider l'attention vers les zones complexes
        complexity_loss = F.mse_loss(pred_attention, complexity_maps)
        total_loss += config.config['model']['attention']['loss_weights'].get('complexity', 0.5) * complexity_loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'diversity_loss': div_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'score_concentration_loss': score_concentration_loss.item(),
            'complexity_loss': complexity_loss.item()
        }
    
    @staticmethod
    def points_to_attention_map(points, scores, size):
        """
        Convertit les points d'attention en carte d'attention pour l'entraînement.
        """
        B, N, _ = points.shape
        H, W = size
        attention_maps = torch.zeros((B, 1, H, W), device=points.device)
        
        # Convertir coordonnées normalisées en pixels
        points = points.clone()
        points[:, :, 0] *= W
        points[:, :, 1] *= H
        
        for b in range(B):
            for i in range(N):
                x, y = points[b, i]
                score = scores[b, i]
                
                # Créer une gaussienne autour du point
                y_grid, x_grid = torch.meshgrid(
                    torch.arange(H, device=points.device),
                    torch.arange(W, device=points.device)
                )
                
                # Rayon d'influence de 128 pixels
                sigma = 128 / 3  # 3 sigma rule
                gaussian = torch.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
                attention_maps[b, 0] += gaussian * score
                
        # Normaliser
        attention_maps = attention_maps / attention_maps.max()
        
        return attention_maps

    def detect_panel_edges(self, x):
        """Détecte les bords des cases de manga"""
        # Convertir en grayscale
        gray = 0.299 * x[:,0] + 0.587 * x[:,1] + 0.114 * x[:,2]
        
        # Appliquer Canny
        sigma = 1.0
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        
        # Gaussian blur avec torchvision
        gaussian = TF.gaussian_blur(
            gray.unsqueeze(1), 
            kernel_size=kernel_size,
            sigma=sigma
        )
        
        # Sobel pour trouver les gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).float()
        
        grad_x = F.conv2d(gaussian, sobel_x.view(1, 1, 3, 3), padding=1)
        grad_y = F.conv2d(gaussian, sobel_y.view(1, 1, 3, 3), padding=1)
        
        # Magnitude et direction
        magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        return magnitude

    def distribute_points(self, points):
        """Force une meilleure distribution des points"""
        batch_size = points.size(0)
        num_points = min(25, points.size(1))  # Limiter à 25 points
        
        # Grille régulière pour initialisation
        grid_size = int(np.sqrt(num_points))
        x = torch.linspace(0.1, 0.9, grid_size)
        y = torch.linspace(0.1, 0.9, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        # S'assurer que nous avons le bon nombre de points
        grid_points = grid_points[:num_points]
        
        # Répéter pour chaque batch
        grid_points = grid_points.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Ajouter du bruit aux points de la grille
        noise = torch.randn_like(grid_points) * 0.05
        points = grid_points + noise
        
        # Clipper entre 0 et 1
        points = torch.clamp(points, 0, 1)
        
        return points

    def filter_attention_points(self, points, scores, min_points=5, min_score_threshold=0.3, max_points=15):
        """
        Filtre les points d'attention selon leur score et les contraintes.
        
        Args:
            points (torch.Tensor): Points d'attention [B, N, 2]
            scores (torch.Tensor): Scores d'importance [B, N]
            min_points (int): Nombre minimum de points à garder
            min_score_threshold (float): Seuil minimum pour le score
            max_points (int): Nombre maximum de points à garder
        """
        batch_size = points.size(0)
        num_points = points.size(1)  # Nombre de points actuels
        
        # Debug
        print(f"Input shapes - Points: {points.shape}, Scores: {scores.shape}")
        
        filtered_points = []
        filtered_scores = []
        
        for b in range(batch_size):
            # Trier les points par score
            current_scores = scores[b, :num_points]  # Prendre seulement les scores valides
            current_points = points[b, :num_points]  # Prendre seulement les points valides
            
            sorted_indices = torch.argsort(current_scores, descending=True)
            sorted_scores = current_scores[sorted_indices]
            sorted_points = current_points[sorted_indices]
            
            # Appliquer le seuil de score
            mask = sorted_scores >= min_score_threshold
            
            # S'assurer d'avoir au moins min_points
            if mask.sum() < min_points:
                mask[:min_points] = True
            
            # Limiter au nombre maximum de points
            max_idx = min(max_points, len(mask))
            mask = mask[:max_idx]
            
            filtered_points.append(sorted_points[mask])
            filtered_scores.append(sorted_scores[mask])
            
            # Debug
            print(f"Batch {b} - Points filtrés: {filtered_points[-1].shape}, Scores filtrés: {filtered_scores[-1].shape}")
        
        # Padding pour avoir des tenseurs de taille uniforme
        max_filtered = max(len(p) for p in filtered_points)
        max_filtered = min(max_filtered, max_points)  # Ne pas dépasser max_points
        
        padded_points = torch.zeros(batch_size, max_filtered, 2, device=points.device)
        padded_scores = torch.zeros(batch_size, max_filtered, device=scores.device)
        
        for b in range(batch_size):
            n = min(len(filtered_points[b]), max_filtered)
            padded_points[b, :n] = filtered_points[b][:n]
            padded_scores[b, :n] = filtered_scores[b][:n]
        
        return padded_points, padded_scores

    def remove_white_margins(self, x):
        """
        Supprime les marges blanches autour de l'image.
        
        Returns:
            - x_cropped: Image recadrée
            - crop_coords: (top, left, bottom, right) coordonnées du recadrage
        """
        # Convertir en niveau de gris si ce n'est pas déjà fait
        if x.size(1) == 3:
            gray = 0.299 * x[:,0] + 0.587 * x[:,1] + 0.114 * x[:,2]
        else:
            gray = x[:,0]
        
        batch_size = x.size(0)
        crops = []
        crop_coords = []
        
        for b in range(batch_size):
            # Trouver les pixels non-blancs (seuil à 0.95 pour tolérer le bruit)
            mask = gray[b] < 0.95
            
            # Trouver les limites des pixels non-blancs
            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)
            
            top, bottom = torch.where(rows)[0][[0, -1]]
            left, right = torch.where(cols)[0][[0, -1]]
            
            # Ajouter une marge de sécurité de 5 pixels
            margin = 5
            top = max(0, top - margin)
            left = max(0, left - margin)
            bottom = min(x.size(-2), bottom + margin)
            right = min(x.size(-1), right + margin)
            
            # Recadrer l'image
            x_cropped = x[b:b+1, :, top:bottom, left:right]
            crops.append(x_cropped)
            crop_coords.append((top, left, bottom, right))
        
        # Reconstruire le batch avec padding pour avoir des tailles uniformes
        max_h = max(crop[0].size(-2) for crop in crops)
        max_w = max(crop[0].size(-1) for crop in crops)
        
        padded_crops = []
        for crop in crops:
            pad_h = max_h - crop.size(-2)
            pad_w = max_w - crop.size(-1)
            padded = F.pad(crop, (0, pad_w, 0, pad_h), value=1.0)  # Padding blanc
            padded_crops.append(padded)
        
        return torch.cat(padded_crops, dim=0), crop_coords

    def adjust_points_to_original(self, points, crop_coords):
        """
        Ajuste les coordonnées des points pour correspondre à l'image d'origine.
        """
        batch_size = points.size(0)
        adjusted_points = points.clone()
        
        for b in range(batch_size):
            top, left, bottom, right = crop_coords[b]
            
            # Convertir les coordonnées normalisées en coordonnées absolues
            h = bottom - top
            w = right - left
            
            # Ajuster les coordonnées x
            adjusted_points[b, :, 0] = adjusted_points[b, :, 0] * w + left
            # Ajuster les coordonnées y
            adjusted_points[b, :, 1] = adjusted_points[b, :, 1] * h + top
            
            # Renormaliser par rapport à la taille originale
            adjusted_points[b, :, 0] /= self.vit.image_size
            adjusted_points[b, :, 1] /= self.vit.image_size
        
        return adjusted_points

    def find_local_maxima(self, attention_map, min_distance=10, threshold=0.5):
        """
        Trouve les maxima locaux dans la carte d'attention.
        
        Args:
            attention_map: Carte d'attention (B, 1, H, W)
            min_distance: Distance minimale entre deux maxima
            threshold: Seuil minimal pour considérer un maximum
            
        Returns:
            maxima_coords: Liste de coordonnées (x, y) des maxima locaux
            maxima_values: Valeurs des maxima
        """
        batch_size = attention_map.size(0)
        maxima_coords = []
        maxima_values = []
        
        for b in range(batch_size):
            # Convertir en numpy pour utiliser peak_local_max
            att_map = attention_map[b, 0].cpu().numpy()
            
            # Trouver les maxima locaux
            from skimage.feature import peak_local_max
            coordinates = peak_local_max(
                att_map,
                min_distance=min_distance,
                threshold_abs=threshold,
                exclude_border=False
            )
            
            # Récupérer les valeurs des maxima
            values = torch.tensor([att_map[y, x] for y, x in coordinates], 
                                device=attention_map.device)
            
            # Convertir en coordonnées (x, y)
            coords = torch.tensor(coordinates[:, [1, 0]], device=attention_map.device)
            
            maxima_coords.append(coords)
            maxima_values.append(values)
        
        return maxima_coords, maxima_values
