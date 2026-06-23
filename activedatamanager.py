"""
DataManager modifications to support ActiveNeRF's incremental view selection.

This module shows how to extend the base DataManager to support:
- Filtering training data by active view indices
- Dynamic dataset updates during training
- Efficient data loading for active learning
"""

from typing import List, Optional, Dict, Any
import torch
from torch.utils.data import Subset
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager


class ActiveNeRFDataManager(VanillaDataManager):
    """
    DataManager that supports active view selection by filtering training data.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_indices: Optional[List[int]] = None
        self._full_train_dataset = None
        
    def setup_train(self):
        """Setup training dataset with support for active view filtering"""
        super().setup_train()
        
        # Store reference to full dataset
        self._full_train_dataset = self.train_dataset
        
    def set_active_indices(self, indices: List[int]) -> None:
        """
        Update the active training views.
        
        Args:
            indices: List of image indices to use for training
        """
        self.active_indices = indices
        
        # Update the dataset to only use active views
        if self._full_train_dataset is not None:
            # Option 1: Use PyTorch Subset (simple but may not work with all datasets)
            # self.train_dataset = Subset(self._full_train_dataset, indices)
            
            # Option 2: If your dataset supports it, update internal indices
            if hasattr(self.train_dataset, 'set_image_indices'):
                self.train_dataset.set_image_indices(indices)
            
            # Option 3: Update the dataparser outputs
            if hasattr(self, 'train_dataparser_outputs'):
                self._filter_dataparser_outputs(indices)
            
            # Recreate dataloader with new dataset
            self._setup_train_dataloader()
    
    def _filter_dataparser_outputs(self, indices: List[int]) -> None:
        """
        Filter dataparser outputs to only include active views.
        
        Args:
            indices: Indices of images to keep
        """
        outputs = self.train_dataparser_outputs
        
        # Filter cameras
        if hasattr(outputs, 'cameras'):
            # Keep only the selected cameras
            original_cameras = outputs.cameras
            filtered_cameras = original_cameras[indices]
            outputs.cameras = filtered_cameras
        
        # Filter image filenames
        if hasattr(outputs, 'image_filenames'):
            original_filenames = outputs.image_filenames
            outputs.image_filenames = [original_filenames[i] for i in indices]
        
        # Filter metadata if present
        if hasattr(outputs, 'metadata'):
            for key, value in outputs.metadata.items():
                if isinstance(value, (list, torch.Tensor)):
                    if isinstance(value, list):
                        outputs.metadata[key] = [value[i] for i in indices]
                    else:
                        outputs.metadata[key] = value[indices]
    
    def _setup_train_dataloader(self) -> None:
        """Recreate training dataloader with current dataset"""
        if hasattr(self, 'train_pixel_sampler'):
            # Recreate pixel sampler if needed
            self.train_pixel_sampler.set_num_rays_per_batch(self.config.train_num_rays_per_batch)
        
        # Recreate dataloader
        # This depends on your specific dataloader setup
        pass
    
    def get_active_cameras(self):
        """Get cameras for currently active views"""
        if self.active_indices is None:
            return self.train_dataset.cameras
        return self.train_dataset.cameras[self.active_indices]
    
    def get_available_cameras(self, exclude_active: bool = True):
        """
        Get cameras for available (non-active) views.
        
        Args:
            exclude_active: If True, exclude currently active cameras
        """
        all_cameras = self._full_train_dataset.cameras
        
        if not exclude_active or self.active_indices is None:
            return all_cameras
        
        # Get indices not in active set
        all_indices = set(range(len(all_cameras)))
        active_set = set(self.active_indices)
        available_indices = list(all_indices - active_set)
        
        return all_cameras[available_indices]


class ActiveNeRFDataset:
    """
    Wrapper for datasets to support active view filtering.
    
    This is useful if you can't modify the DataManager directly.
    """
    
    def __init__(self, base_dataset, active_indices: Optional[List[int]] = None):
        self.base_dataset = base_dataset
        self.active_indices = active_indices or list(range(len(base_dataset)))
        
    def set_image_indices(self, indices: List[int]) -> None:
        """Update which images are active"""
        self.active_indices = indices
    
    def __len__(self):
        return len(self.active_indices)
    
    def __getitem__(self, idx):
        # Map from active index to actual dataset index
        actual_idx = self.active_indices[idx]
        return self.base_dataset[actual_idx]
    
    @property
    def cameras(self):
        """Get cameras for active views"""
        if hasattr(self.base_dataset, 'cameras'):
            all_cameras = self.base_dataset.cameras
            return all_cameras[self.active_indices]
        return None


# Example usage in config
"""
To use ActiveNeRF, update your config:

pipeline:
  datamanager:
    _target: path.to.ActiveNeRFDataManager
    # ... other datamanager config ...

trainer:
  _target: path.to.ActiveNeRFTrainer
  initial_views: 4
  views_per_iteration: 1
  steps_per_active_selection: 5000
  uncertainty_method: "entropy"
  candidate_views_sampling: 100
  # ... other trainer config ...
"""


# Utility functions for active learning

def compute_view_coverage(cameras, resolution: int = 64) -> torch.Tensor:
    """
    Compute spatial coverage of views in scene.
    
    Args:
        cameras: Camera objects
        resolution: Grid resolution for coverage computation
        
    Returns:
        Coverage heatmap as tensor
    """
    # Create voxel grid
    coverage = torch.zeros((resolution, resolution, resolution))
    
    for camera in cameras:
        # Get camera position and direction
        position = camera.camera_to_worlds[:3, 3]
        direction = camera.camera_to_worlds[:3, 2]
        
        # Mark voxels along ray (simplified)
        # In practice, you'd want to raymarch through the scene
        grid_pos = ((position + 1) / 2 * resolution).long()
        grid_pos = torch.clamp(grid_pos, 0, resolution - 1)
        
        coverage[grid_pos[0], grid_pos[1], grid_pos[2]] += 1
    
    return coverage


def diversity_based_selection(
    candidate_cameras, 
    active_cameras, 
    num_select: int
) -> List[int]:
    """
    Select diverse views based on camera pose diversity.
    
    Args:
        candidate_cameras: Candidate camera poses
        active_cameras: Currently active camera poses
        num_select: Number of views to select
        
    Returns:
        Indices of selected cameras
    """
    selected = []
    
    # Compute distances from each candidate to active cameras
    for i, cand_cam in enumerate(candidate_cameras):
        cand_pos = cand_cam.camera_to_worlds[:3, 3]
        
        # Distance to nearest active camera
        min_dist = float('inf')
        for act_cam in active_cameras:
            act_pos = act_cam.camera_to_worlds[:3, 3]
            dist = torch.norm(cand_pos - act_pos)
            min_dist = min(min_dist, dist.item())
        
        selected.append((i, min_dist))
    
    # Select cameras with maximum distance to active set
    selected.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in selected[:num_select]]