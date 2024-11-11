from typing import Dict, List
import pandas as pd
import os
import json

class MetricsAnalyzer:
    """
    Classe utilitaire pour analyser les métriques stockées.
    """
    def __init__(self, metrics_csv_path: str):
        self.metrics_df = pd.read_csv(metrics_csv_path)
        
    def get_best_results(self, metric_name: str, n: int = 10) -> pd.DataFrame:
        """
        Retourne les N meilleurs résultats pour une métrique donnée.
        """
        return self.metrics_df.nlargest(n, metric_name)
    
    def get_worst_results(self, metric_name: str, n: int = 10) -> pd.DataFrame:
        """
        Retourne les N pires résultats pour une métrique donnée.
        """
        return self.metrics_df.nsmallest(n, metric_name)
    
    def export_summary(self, output_path: str):
        """
        Exporte un résumé des métriques au format JSON.
        """
        summary = {
            'total_images': len(self.metrics_df),
            'metrics_summary': {}
        }
        
        # Pour chaque métrique numérique
        for column in self.metrics_df.select_dtypes(include=['float64', 'int64']).columns:
            summary['metrics_summary'][column] = {
                'mean': float(self.metrics_df[column].mean()),
                'std': float(self.metrics_df[column].std()),
                'min': float(self.metrics_df[column].min()),
                'max': float(self.metrics_df[column].max()),
                'median': float(self.metrics_df[column].median())
            }
            
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)