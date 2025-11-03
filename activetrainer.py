from dataclasses import dataclass, field
from typing import DefaultDict        # Para type hints
from collections import defaultdict   # Para crear instancias
from typing import Type, Optional, Dict, List, Literal
from pathlib import Path
import torch
import numpy as np
from threading import Lock
import time

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import writer, profiler
from nerfstudio.utils.writer import EventName, TimeWriter


@dataclass
class ActiveNeRFTrainerConfig(TrainerConfig):
    """Configuration for ActiveNeRF training"""
    
    _target: Type = field(default_factory=lambda: ActiveNeRFTrainer)
    
    # ActiveNeRF specific parameters
    initial_views: int = 4
    """Number of initial views to start training with"""
    
    views_per_iteration: int = 1
    """Number of views to add in each active selection iteration"""
    
    steps_per_active_selection: int = 500
    """Number of training steps between active view selections"""
    
    uncertainty_method: Literal["entropy", "variance", "ensemble"] = "entropy"
    """Method to compute uncertainty for view selection"""
    
    candidate_views_sampling: int = 100
    """Number of candidate views to sample for selection"""
    
    max_views: Optional[int] = None
    """Maximum number of views to use (None = use all available)"""
    
    uncertainty_grid_resolution: int = 128
    """Resolution of the grid for uncertainty estimation"""


class ActiveNeRFTrainer(Trainer):
    """
    ActiveNeRF Trainer that implements active learning for view selection.
    
    Extends the base Trainer to add:
    - Incremental view addition
    - Uncertainty-based view selection
    - Active learning loop
    """
    
    def __init__(self, config: ActiveNeRFTrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)
        self.config: ActiveNeRFTrainerConfig = config
        
        # ActiveNeRF specific attributes
        self.active_views: List[int] = []
        self.available_views: List[int] = []
        self.uncertainty_history: List[float] = []
        
    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup with ActiveNeRF modifications"""
        super().setup(test_mode=test_mode)
        
        # Initialize view selection
        self._initialize_active_views()
        
    def _initialize_active_views(self) -> None:
        """Initialize the active learning process by selecting initial views"""
        total_views = len(self.pipeline.datamanager.train_dataset)
        all_indices = list(range(total_views))
        
        # Randomly select initial views
        np.random.shuffle(all_indices)
        self.active_views = all_indices[:self.config.initial_views]
        self.available_views = all_indices[self.config.initial_views:]
        
        # Update datamanager to only use active views
        self._update_active_dataset()
        
        writer.put_scalar(
            name="ActiveNeRF/NumActiveViews", 
            scalar=len(self.active_views), 
            step=0
        )
        
    def _update_active_dataset(self) -> None:
        """Update the training dataset to only include active views"""
        # This will depend on your specific datamanager implementation
        # You may need to modify the datamanager to support view filtering
        if hasattr(self.pipeline.datamanager, 'set_active_indices'):
            self.pipeline.datamanager.set_active_indices(self.active_views)
        else:
            # Alternative: Create a subset of the dataset
            self.pipeline.datamanager.train_dataset.set_active_indices(self.active_views)
    
    def train(self) -> None:
        """Modified training loop with active view selection"""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatasetInputs"
        
        if hasattr(self.pipeline.datamanager, "train_dataparser_outputs"):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )
        
        self._init_viewer_state()
        
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations - self._start_step
            self.stop_training = False
            
            for step in range(self._start_step, self._start_step + num_iterations):
                self.step = step
                
                if self.stop_training:
                    break
                    
                # Check if we should perform active view selection
                if self._should_perform_active_selection(step):
                    self._perform_active_selection(step)
                
                while self.training_state == "paused":
                    if self.stop_training:
                        self._after_train()
                        return
                    time.sleep(0.01)
                
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()
                        
                        # Training callbacks before iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )
                        
                        # Training iteration
                        loss, loss_dict, metrics_dict = self.train_iteration(step)
                        
                        # Training callbacks after iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )
                
                # Update timing metrics
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )
                
                self._update_viewer_state(step)
                
                # Logging
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    writer.put_scalar(
                        name="GPU Memory (MB)", 
                        scalar=torch.cuda.max_memory_allocated() / (1024**2), 
                        step=step
                    )
                
                # Evaluation
                if self.pipeline.datamanager.eval_dataset:
                    with self.train_lock:
                        self.eval_iteration(step)
                
                # Checkpoint saving
                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)
                
                writer.write_out_storage()
        
        self._after_train()
    
    def _should_perform_active_selection(self, step: int) -> bool:
        """Check if we should perform active view selection at this step"""
        if len(self.available_views) == 0:
            return False
        
        if self.config.max_views is not None and len(self.active_views) >= self.config.max_views:
            return False
        
        return step > 0 and step % self.config.steps_per_active_selection == 0
    
    def _perform_active_selection(self, step: int) -> None:
        """Perform active view selection and add new views to training set"""
        writer.put_scalar(
            name="ActiveNeRF/SelectionEvent", 
            scalar=1, 
            step=step
        )
        
        # Sample candidate views
        num_candidates = min(self.config.candidate_views_sampling, len(self.available_views))
        candidate_indices = np.random.choice(
            self.available_views, 
            size=num_candidates, 
            replace=False
        )
        
        # Compute uncertainty for each candidate
        uncertainties = self._compute_view_uncertainties(candidate_indices)
        
        # Select views with highest uncertainty
        num_to_select = min(self.config.views_per_iteration, len(self.available_views))
        selected_idx = np.argsort(uncertainties)[-num_to_select:]
        selected_views = [candidate_indices[i] for i in selected_idx]
        
        # Add selected views to active set
        self.active_views.extend(selected_views)
        for view in selected_views:
            self.available_views.remove(view)
        
        # Update dataset
        self._update_active_dataset()
        
        # Log statistics
        avg_uncertainty = np.mean(uncertainties)
        max_uncertainty = np.max(uncertainties)
        self.uncertainty_history.append(max_uncertainty)
        
        writer.put_scalar(
            name="ActiveNeRF/NumActiveViews", 
            scalar=len(self.active_views), 
            step=step
        )
        writer.put_scalar(
            name="ActiveNeRF/AvgUncertainty", 
            scalar=avg_uncertainty, 
            step=step
        )
        writer.put_scalar(
            name="ActiveNeRF/MaxUncertainty", 
            scalar=max_uncertainty, 
            step=step
        )
        
        print(f"[ActiveNeRF] Step {step}: Added {num_to_select} views. "
              f"Total active views: {len(self.active_views)}/{len(self.active_views) + len(self.available_views)}")
    
    def _compute_view_uncertainties(self, candidate_indices: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty scores for candidate views.
        
        Args:
            candidate_indices: Indices of candidate views to evaluate
            
        Returns:
            Array of uncertainty scores for each candidate
        """
        self.pipeline.eval()
        uncertainties = []
        
        with torch.no_grad():
            for idx in candidate_indices:
                # Get camera for this view
                camera = self.pipeline.datamanager.train_dataset.cameras[int(idx)]
                
                if self.config.uncertainty_method == "entropy":
                    uncertainty = self._compute_entropy_uncertainty(camera)
                elif self.config.uncertainty_method == "variance":
                    uncertainty = self._compute_variance_uncertainty(camera)
                elif self.config.uncertainty_method == "ensemble":
                    uncertainty = self._compute_ensemble_uncertainty(camera)
                else:
                    raise ValueError(f"Unknown uncertainty method: {self.config.uncertainty_method}")
                
                uncertainties.append(uncertainty)
        
        self.pipeline.train()
        return np.array(uncertainties)
    
    def _compute_entropy_uncertainty(self, camera) -> float:
        """Compute entropy-based uncertainty for a camera view"""
        # Render the view
        camera_ray_bundle = camera.generate_rays(camera_indices=0)
        
        # Get model outputs
        outputs = self.pipeline.model(camera_ray_bundle)
        
        # Compute entropy from density/occupancy
        if "density" in outputs:
            density = outputs["density"]
            # Normalize to probability
            prob = torch.sigmoid(density)
            entropy = -(prob * torch.log(prob + 1e-10) + (1 - prob) * torch.log(1 - prob + 1e-10))
            uncertainty = entropy.mean().item()
        else:
            # Fallback: use RGB variance as proxy
            rgb = outputs["rgb"]
            uncertainty = rgb.var(dim=-1).mean().item()
        
        return uncertainty
    
    def _compute_variance_uncertainty(self, camera) -> float:
        """Compute variance-based uncertainty (requires multiple forward passes)"""
        num_samples = 5
        camera_ray_bundle = camera.generate_rays(camera_indices=0)
        
        rgb_samples = []
        for _ in range(num_samples):
            outputs = self.pipeline.model(camera_ray_bundle)
            rgb_samples.append(outputs["rgb"])
        
        rgb_stack = torch.stack(rgb_samples, dim=0)
        variance = rgb_stack.var(dim=0).mean().item()
        
        return variance
    
    def _compute_ensemble_uncertainty(self, camera) -> float:
        """Compute ensemble-based uncertainty (requires ensemble model)"""
        # This requires a model that supports ensemble predictions
        # Placeholder implementation
        return self._compute_variance_uncertainty(camera)
    
    def save_checkpoint(self, step: int) -> None:
        """Save checkpoint with active view information"""
        super().save_checkpoint(step)
        
        # Save active view indices
        active_views_path = self.checkpoint_dir / f"active_views-{step:09d}.npy"
        np.save(active_views_path, np.array(self.active_views))
        
        # Save uncertainty history
        uncertainty_path = self.checkpoint_dir / f"uncertainty_history-{step:09d}.npy"
        np.save(uncertainty_path, np.array(self.uncertainty_history))


# Helper function (if not already imported)
def step_check(step: int, step_size: int, run_at_zero: bool = False) -> bool:
    """Check if we should run at this step"""
    if run_at_zero and step == 0:
        return True
    return step % step_size == 0