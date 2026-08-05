"""On-demand deterministic PC-NSS dataset."""

from torch.utils.data import Dataset

from multisource_doa.config import ExperimentConfig, SplitName
from multisource_doa.data.simulator import DOASample, generate_two_source_sample


class PCNSSDataset(Dataset):
    def __init__(self, split: SplitName, config: ExperimentConfig):
        self.split = SplitName(split)
        self.config = config
        self.config.split.require_access(self.split)
        self.split_seed = int(self.config.split.seeds[self.split])
        self.size = int(self.config.split.sizes[self.split])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> DOASample:
        if not 0 <= index < self.size:
            raise IndexError(f"sample index {index} outside [0, {self.size})")
        return generate_two_source_sample(
            self.config,
            split_seed=self.split_seed,
            index=index,
        )
