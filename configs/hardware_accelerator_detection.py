import tensorflow as tf
import os

def detect_hardware():
    """
    Détecte et configure le hardware disponible (CPU/GPU/TPU)
    Retourne la stratégie appropriée et un résumé du hardware
    """
    print("\n🔍 Détection du hardware...")

    # Essai de détection TPU
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        print("✓ TPU détecté et initialisé:")
        print(f"  - Nombre de TPUs: {strategy.num_replicas_in_sync}")
        return strategy, "TPU"
    except:
        print("  Pas de TPU détecté, recherche de GPU...")

    # Détection GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configuration de la mémoire GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Configuration multi-GPU si disponible
            strategy = tf.distribute.MirroredStrategy()

            # Informations détaillées sur les GPUs
            print("✓ GPU(s) détecté(s) et configuré(s):")
            print(f"  - Nombre de GPUs: {strategy.num_replicas_in_sync}")
            for gpu in gpus:
                desc = tf.config.experimental.get_device_details(gpu)
                if desc:
                    print(f"  - {gpu.device_type}: {desc.get('device_name', gpu.name)}")
                    if 'compute_capability' in desc:
                        print(f"    Compute capability: {desc['compute_capability'][0]}.{desc['compute_capability'][1]}")
                else:
                    print(f"  - {gpu.device_type}: {gpu.name}")

            # Informations sur la mémoire GPU si disponible
            try:
                memory_info = []
                for gpu in gpus:
                    memory_limit = tf.config.experimental.get_memory_info(f'/device:GPU:{gpus.index(gpu)}')
                    total_memory = memory_limit['current'] / (1024**3)  # Conversion en GB
                    memory_info.append(f"    Mémoire disponible: {total_memory:.2f} GB")
                if memory_info:
                    print("\n  Informations mémoire:")
                    print("\n".join(memory_info))
            except:
                pass

            return strategy, "GPU"

        except RuntimeError as e:
            print(f"⚠️ Erreur lors de la configuration GPU: {e}")
            print("  Utilisation du CPU à la place")
            strategy = tf.distribute.get_strategy()
            return strategy, "CPU"
    else:
        print("  Pas de GPU détecté, utilisation du CPU")
        strategy = tf.distribute.get_strategy()
        return strategy, "CPU"

def print_hardware_info():
    """
    Affiche des informations détaillées sur le hardware et l'environnement
    """
    print("\n💻 Configuration système:")
    print(f"  - TensorFlow version: {tf.__version__}")
    print(f"  - Python version: {os.sys.version.split()[0]}")

    # Informations sur le CPU
    try:
        import cpuinfo
        cpu_info = cpuinfo.get_cpu_info()
        print(f"  - CPU: {cpu_info['brand_raw']}")
        print(f"  - Nombre de cœurs CPU: {cpu_info['count']}")
    except:
        print("  - Impossible d'obtenir les informations détaillées du CPU")

    # Informations sur la mémoire système
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"  - Mémoire système totale: {memory.total / (1024**3):.1f} GB")
        print(f"  - Mémoire système disponible: {memory.available / (1024**3):.1f} GB")
    except:
        print("  - Impossible d'obtenir les informations de mémoire système")

def setup_hardware():
    """
    Configure le hardware et retourne la stratégie appropriée
    """
    print("\n🚀 Configuration du hardware d'accélération...")

    # Activation du XLA JIT si possible
    if os.environ.get('XLA_FLAGS') is None:
        os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda/'
    tf.config.optimizer.set_jit(True)
    print("✓ XLA JIT activé")

    # Configuration de la précision mixte
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✓ Précision mixte activée (float16/float32)")

    # Détection et configuration du hardware
    strategy, hardware_type = detect_hardware()

    # Affichage des informations système
    print_hardware_info()

    return strategy, hardware_type

