from collections import OrderedDict
from typing import Dict, Union
from nerfstudio.configs.external_methods import ExternalMethodDummyTrainerConfig, get_external_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.engine.optimizers import RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.configs.base_config import ViewerConfig
from activenerf_config import ActiveNeRFModelConfig


from activedatamanager import ActiveNeRFDataManager
from activetrainer import ActiveNeRFTrainerConfig

method_configs: Dict[str, Union[ActiveNeRFTrainerConfig, ExternalMethodDummyTrainerConfig]] = {}

method_configs["activenerf"] = ActiveNeRFTrainerConfig(
    method_name="activenerf_start1view",
    
    # === Parámetros de ActiveNeRF (nuevos) ===
    initial_views=1,                        # Número de vistas iniciales
    views_per_iteration=1,                  # Vistas a añadir por iteración
    steps_per_active_selection=20000,        # Cada cuántos steps seleccionar
    uncertainty_method="entropy",           # "entropy", "variance", "ensemble"
    candidate_views_sampling=15,           # Candidatos a evaluar
    max_views=None,                         # None = usar todas disponibles
    ray_sample_min = 4096,                  # Numero minimo de rayos
    uncertainty_grid_resolution=128,        # Resolución para cálculo de incertidumbre
    

    # === Pipeline Configuration ===
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=ActiveNeRFDataManager,  # <-- Usar DataManager customizado
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096, # 4096
        ),
        model=ActiveNeRFModelConfig(
            beta_min=0.01,
            w=0.01,
            use_uncertainty=True,           # <-- ACTIVADO para ActiveNeRF
            # Parámetros específicos para incertidumbre
            enable_uncertainty_estimation=True,
            uncertainty_method="density_fine",  # "density" o "rgb_variance"
            use_uncertainty_loss = False,
            enable_entropy_regularization = False, 
        ),
    ),
    
    # === Optimizers ===
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(
                lr=5e-4,
                eps=1e-15,
            ),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-6,
                max_steps=200000,
            ),
        },
        # Optimizador adicional para parámetros de incertidumbre si es necesario
        "uncertainty_head": {
            "optimizer": RAdamOptimizerConfig(
                lr=1e-4,
                eps=1e-15,
            ),
            "scheduler": None,
        }
    },
    
    # === Training Parameters ===
    max_num_iterations=200000,              # Total de iteraciones
    steps_per_save=50000,                   # Guardar checkpoints
    steps_per_eval_batch=1500,
    steps_per_eval_image=1500,
    steps_per_eval_all_images=50000,
    
    # === Mixed Precision ===
    mixed_precision=True,
    use_grad_scaler=True,
    
    # === Viewer Configuration ===
    viewer=ViewerConfig(
        num_rays_per_chunk=1 << 15,
        quit_on_train_completion=False,
    ),
    
    # === Logging ===
    vis="wandb",                      # "tensorboard", "wandb", o "viewer+tensorboard"
    log_gradients=False,                    # True si quieres loggear gradientes
    
    # === Checkpoint/Resume ===
    save_only_latest_checkpoint=False,      # False para guardar todos
    load_dir=None,                          # Path si quieres continuar entrenamiento
    load_step=None,
    load_checkpoint=None,
)


# ===== Configuración Alternativa: ActiveNeRF con Varianza =====
method_configs["activenerf-variance"] = ActiveNeRFTrainerConfig(
    method_name="activenerf-variance",
    
    # ActiveNeRF con método de varianza (más robusto pero más lento)
    initial_views=4,
    views_per_iteration=2,                  # Añadir 2 vistas por vez
    steps_per_active_selection=50,        # Más frecuente
    uncertainty_method="variance",          # Requiere múltiples forward passes
    candidate_views_sampling=10,            # Menos candidatos (es más lento)
    ray_sample_min = 4096,   # Numero minimo de rayos

    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=ActiveNeRFDataManager,
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=4096,
        ),
        model=ActiveNeRFModelConfig(
            beta_min=0.01,
            w=0.01,
            use_uncertainty=True,
            enable_uncertainty_estimation=True,
            uncertainty_method="monte_carlo_dropout",  # Para varianza
        ),
    ),
    
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4),
            "scheduler": None,
        }
    },
    
    max_num_iterations=200000,
    mixed_precision=True,
)


# ===== Configuración: ActiveNeRF con más vistas iniciales =====
method_configs["activenerf-conservative"] = ActiveNeRFTrainerConfig(
    method_name="activenerf-conservative",
    
    # Más vistas iniciales = más conservador pero más estable
    initial_views=8,                        # Más vistas al inicio
    views_per_iteration=1,
    steps_per_active_selection=50,        # Menos frecuente
    uncertainty_method="entropy",
    candidate_views_sampling=10,
    max_views=50,                           # Límite de vistas totales
    ray_sample_min = 4096,   # Numero minimo de rayos

    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            _target=ActiveNeRFDataManager,
            dataparser=BlenderDataParserConfig(),
            train_num_rays_per_batch=8192,  # Más rays por batch
        ),
        model=ActiveNeRFModelConfig(
            beta_min=0.01,
            w=0.01,
            use_uncertainty=True,
            enable_uncertainty_estimation=True,
        ),
    ),
    
    optimizers={
        "fields": {
            "optimizer": RAdamOptimizerConfig(lr=5e-4),
            "scheduler": None,
        }
    },
    
    max_num_iterations=300000,              # Más iteraciones
    steps_per_save=15000,
    mixed_precision=True,
)


# ===== Helper: Función para crear config personalizado =====
def create_activenerf_config(
    initial_views: int = 4,
    views_per_iteration: int = 1,
    steps_per_selection: int = 500,
    uncertainty_method: str = "entropy",
    max_iterations: int = 200000,
    ray_sample_min = 4096, 
    **kwargs
) -> ActiveNeRFTrainerConfig:
    """
    Helper para crear configuraciones personalizadas de ActiveNeRF
    
    Args:
        initial_views: Número de vistas iniciales
        views_per_iteration: Vistas a añadir cada vez
        steps_per_selection: Frecuencia de selección activa
        uncertainty_method: "entropy", "variance", o "ensemble"
        max_iterations: Iteraciones totales de entrenamiento
        **kwargs: Otros parámetros a sobreescribir
    """
    config = ActiveNeRFTrainerConfig(
        method_name="activenerf-custom",
        initial_views=initial_views,
        views_per_iteration=views_per_iteration,
        steps_per_active_selection=steps_per_selection,
        uncertainty_method=uncertainty_method,
        max_num_iterations=max_iterations,
        ray_sample_min = ray_sample_min, 

        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                _target=ActiveNeRFDataManager,
                dataparser=BlenderDataParserConfig(),
            ),
            model=ActiveNeRFModelConfig(
                beta_min=0.01,
                w=0.01,
                use_uncertainty=True,
            ),
        ),
        
        optimizers={
            "fields": {
                "optimizer": RAdamOptimizerConfig(lr=5e-4),
                "scheduler": None,
            }
        },
        
        mixed_precision=True,
    )
    
    # Aplicar overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

