from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from awex import logging

logger = logging.getLogger(__name__)


class Engine(ABC):
    
    def __init__(self, hf_config):
        self.global_step = -1
        self.hf_config = hf_config

    @property
    @abstractmethod
    def engine_name(self):
        pass
    
    def hf_config(self):
        return self.hf_config

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the engine with configuration."""
        pass

    def set_global_step(self, global_step: int):
        logger.info(f"set global step from {self.global_step} to {global_step}")
        self.global_step = global_step

    @abstractmethod
    def release_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Release memory occupied by the engine."""
        pass

    @abstractmethod
    def resume_memory_occupation(self, tags: Optional[List[str]] = None) -> None:
        """Resume memory occupation for the engine."""
        pass


class InferenceEngine(Engine):
    @property
    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def update_weights_from_disk(
        self, model_path: str, load_format: Optional[str] = None
    ):
        """Update weights from disk for inference engine."""
        pass

    @abstractmethod
    def update_weights(self, **kwargs):
        """Update weights for inference engine."""
        pass
    

class TrainingEngine(Engine):

    @abstractmethod
    def save_hf_checkpoint(self, path: str):
        """Save model checkpoint."""
        pass

    @abstractmethod
    def write_weights(self, **kwargs):
        """Write weights for training engine."""
        pass

    @abstractmethod
    def release_grad_memory(self, empty_cache=True):
        pass