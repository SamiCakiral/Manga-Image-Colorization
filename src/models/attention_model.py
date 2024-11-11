import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.config import config

class AttentionPointsModel(nn.Module):
    """Cette classe est un modèle d'attention qui prédit des points clés pour guider le découpage intelligent. 
    Elle est basée sur le modèle ViT-B/16 de PyTorch.
    Et elle utilise une tête d'attention pour prédire les points d'attention Avec une gaussienne autour du point.
    ViT-B/16 est un modèle de vision transformer pré-entraîné sur ImageNet qui est utilisé pour extraire des features.››
    """
    def __init__(self, max_points=config.max_attention_points):
        """
        Initialise le modèle d'attention qui prédit des points clés.
        
        Arguments:
        - max_points: Nombre maximum de points d'attention à prédire
        """
        super(AttentionPointsModel, self).__init__()
        
        # Vision Transformer modifié pour prédire des points
        self.vit = models.vit_b_16(weights='IMAGENET1K_V1' if config.vit_pretrained else None)
        
        # Remplacer la tête de classification par notre tête d'attention
        self.attention_head = nn.Sequential(
            nn.Linear(config.vit_feature_size, config.attention_head_hidden),
            nn.ReLU(),
            nn.Linear(config.attention_head_hidden, max_points * 3)  # (x, y, importance_score) pour chaque point
        )
        
        self.max_points = max_points
        
    def forward(self, x):
        """
        Prédit les points d'attention pour l'image.
        
        Returns:
        - points: tensor de forme (batch_size, max_points, 2) pour les coordonnées x,y
        - scores: tensor de forme (batch_size, max_points) pour les scores d'importance
        """
        # Adapter l'image en entrée
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Obtenir les features du ViT
        features = self.vit.forward_features(x)
        
        # Prédire les points et scores
        output = self.attention_head(features[:, 0])  # Utiliser le token [CLS]
        output = output.view(-1, self.max_points, 3)
        
        # Séparer coordonnées et scores
        points = output[:, :, :2]  # x,y coordonnées
        scores = output[:, :, 2]   # scores d'importance
        
        # Normaliser les coordonnées entre 0 et 1
        points = torch.sigmoid(points)
        # Normaliser les scores
        scores = torch.sigmoid(scores)
        
        return points, scores
        
    def get_attention_patches(self, image, points, scores, patch_size=config.patch_size):
        """
        Génère des patches de 256x256 autour des points d'attention.
        
        Arguments:
        - image: tensor de forme (B, C, H, W)
        - points: tensor de forme (B, N, 2) avec coordonnées normalisées
        - scores: tensor de forme (B, N) avec scores d'importance
        - patch_size: taille des patches à extraire
        
        Returns:
        - patches: liste des patches extraits
        - patch_coords: coordonnées des patches dans l'image originale
        """
        B, C, H, W = image.shape
        patches = []
        patch_coords = []
        radius = patch_size // 2
        
        # Convertir les coordonnées normalisées en pixels
        points = points.clone()
        points[:, :, 0] *= W
        points[:, :, 1] *= H
        
        # Pour chaque image dans le batch
        for b in range(B):
            img_patches = []
            img_coords = []
            
            # Trier les points par score d'importance
            _, indices = scores[b].sort(descending=True)
            sorted_points = points[b][indices]
            
            for point in sorted_points:
                x, y = point
                
                # Calculer les coordonnées du patch
                left = max(0, int(x - radius))
                top = max(0, int(y - radius))
                right = min(W, left + patch_size)
                bottom = min(H, top + patch_size)
                
                # Extraire le patch
                patch = image[b:b+1, :, top:bottom, left:right]
                
                # Padding si nécessaire
                if patch.shape[2:] != (patch_size, patch_size):
                    pad_h = patch_size - patch.shape[2]
                    pad_w = patch_size - patch.shape[3]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h))
                
                img_patches.append(patch)
                img_coords.append((left, top, right, bottom))
            
            patches.append(img_patches)
            patch_coords.append(img_coords)
            
        return patches, patch_coords
    
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
        Une étape d'entraînement.
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
        
        # Calculer la perte
        loss = criterion(pred_attention, complexity_maps)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
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