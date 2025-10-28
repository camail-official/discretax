"""Datasets for examples."""

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

logger = logging.getLogger(__name__)


class MNISTSeq(VisionDataset):
    """Sequential MNIST handwriting dataset.

    This dataset contains handwritten digits from MNIST represented as sequences of pen strokes.
    Each sample is a sequence of (dx, dy, stroke_end, digit_end) features representing the
    pen movement needed to draw the digit.

    The dataset is from Edwin de Jong's MNIST sequence data:
    https://edwin-de-jong.github.io/blog/mnist-sequence-data/

    Features:
        - dx: Change in x coordinate (pen movement)
        - dy: Change in y coordinate (pen movement)
        - stroke_end: Binary flag indicating end of a stroke
        - digit_end: Binary flag indicating end of the digit

    Args:
        root: Root directory where dataset will be stored (str or Path).
        train: If True, creates dataset from training set, otherwise from test set.
        download: If True, downloads the dataset from the internet.
        transform: Optional transform to be applied on a sample.
        target_transform: Optional transform to be applied on the target.
        pad_to: Maximum sequence length (sequences are padded/truncated to this length).
        train_val_split: Fraction of training data to use for training (rest is validation).
        stage: Split to use - "train", "val", or "test". If None, inferred from `train` arg.

    Raises:
        ValueError: If stage is invalid.
        RuntimeError: If dataset not found and download is False.

    Example:
        >>> dataset = MNISTSeq(root="./data", train=True, download=True)
        >>> x, y = dataset[0]  # Get first sample
        >>> print(x.shape)  # (128, 4) - sequence of (dx, dy, stroke_end, digit_end)
        >>> print(y)  # 0-9 digit label
    """

    # Official dataset source (from Edwin D. de Jong's blog)
    # Download page: https://edwin-de-jong.github.io/blog/mnist-sequence-data/
    # GitHub repository: https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data
    mirrors = [
        "https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/raw/master/sequences.tar.gz"
    ]

    # Class labels for digits
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        pad_to: int = 128,
        train_val_split: float = 0.8,
        stage: str | None = None,
    ):
        # Initialize VisionDataset (handles root, transform, target_transform)
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Store configuration
        self.pad_to = pad_to
        self.train_val_split = train_val_split

        # Determine which split to use
        self.stage = stage if stage is not None else ("test" if not train else "train")
        if self.stage not in ["train", "val", "test"]:
            raise ValueError(f"Invalid stage: {self.stage}. Must be 'train', 'val', or 'test'")

        # Download dataset if requested
        if download:
            self.download()

        # Preprocess raw data if not already done
        if not self._check_processed():
            if not self._check_raw():
                raise RuntimeError("Dataset not found. Use download=True to download it.")
            self._preprocess()

        # Load the appropriate data split
        self.data, self.labels = self._load_dataset()

    @property
    def raw_folder(self) -> Path:
        """Path to the folder containing raw data files.

        Returns:
            Path to raw data directory.
        """
        # The tar.gz extracts to raw/sequences/ containing the data files
        return Path(self.root) / self.__class__.__name__ / "raw" / "sequences"

    @property
    def processed_folder(self) -> Path:
        """Path to the folder containing preprocessed data files.

        Returns:
            Path to processed data directory.
        """
        return Path(self.root) / self.__class__.__name__ / "processed"

    @property
    def num_classes(self) -> int:
        """Number of target classes.

        Returns:
            Number of digit classes (10 for MNIST digits 0-9).
        """
        return len(self.classes)

    @property
    def input_dim(self) -> int:
        """Number of input features per timestep.

        Returns:
            Number of features (4: dx, dy, stroke_end, digit_end).
        """
        return 4

    @property
    def class_to_idx(self) -> dict[str, int]:
        """Mapping from class name to class index.

        Returns:
            Dictionary mapping class names to their integer indices.
        """
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_raw(self) -> bool:
        """Check if raw data files exist.

        Returns:
            True if raw data exists, False otherwise.
        """
        return (
            (self.raw_folder / "trainlabels.txt").exists()
            and (self.raw_folder / "testlabels.txt").exists()
            and len(list(self.raw_folder.glob("trainimg-*-inputdata.txt"))) > 0
            and len(list(self.raw_folder.glob("testimg-*-inputdata.txt"))) > 0
        )

    def _check_processed(self) -> bool:
        """Check if processed data files exist.

        Returns:
            True if processed data exists, False otherwise.
        """
        return (self.processed_folder / "train_data.npy").exists() and (
            self.processed_folder / "test_data.npy"
        ).exists()

    def download(self) -> None:
        """Download the MNIST sequence dataset from the internet.

        The dataset will be downloaded to the `raw_folder` directory.
        If the data already exists, this method does nothing.

        Raises:
            RuntimeError: If download fails from all mirror URLs.
        """
        if self._check_raw():
            logger.info("Raw data already exists. Skipping download.")
            return

        # Extract to raw/ directory (tar.gz contains sequences/ folder)
        raw_parent = self.raw_folder.parent
        raw_parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading MNIST sequence data...")

        # Try each mirror URL until one succeeds
        for mirror_url in self.mirrors:
            try:
                logger.debug(f"Attempting download from: {mirror_url}")
                download_and_extract_archive(
                    mirror_url,
                    download_root=str(raw_parent),
                    extract_root=str(raw_parent),
                )
                logger.info("Download successful!")
                return
            except Exception as e:
                logger.warning(f"Failed to download from {mirror_url}: {e}")

        # If we get here, all mirrors failed
        raise RuntimeError(
            "Failed to download dataset from all mirrors. "
            "Please download manually from: "
            "https://edwin-de-jong.github.io/blog/mnist-sequence-data/"
        )

    def _preprocess(self) -> None:
        """Preprocess raw text files into numpy arrays.

        This method:
        1. Reads all individual text files for each sample
        2. Pads/truncates sequences to a fixed length
        3. Saves as single .npy files for faster loading

        The processed files are saved to `processed_folder`.

        Raises:
            FileNotFoundError: If no files matching the pattern are found in the raw folder.
        """
        logger.info("Preprocessing MNIST sequence data...")
        self.processed_folder.mkdir(parents=True, exist_ok=True)

        # Process both training and test sets
        for split in ["train", "test"]:
            # Find all input data files for this split
            pattern = f"{split}img-*-inputdata.txt"
            files = list(self.raw_folder.glob(pattern))

            if not files:
                raise FileNotFoundError(
                    f"No files matching '{pattern}' found in {self.raw_folder}"
                )

            # Sort files by their numerical index for consistent ordering
            files.sort(key=lambda x: int(x.stem.split("-")[1]))

            # Load and process each file
            data_list = []
            for filepath in files:
                # Load sequence data (shape: [seq_len, 4])
                sequence = np.loadtxt(filepath)

                # Pad or truncate to fixed length
                if sequence.shape[0] < self.pad_to:
                    # Pad with zeros if sequence is too short
                    padding = ((0, self.pad_to - sequence.shape[0]), (0, 0))
                    sequence = np.pad(sequence, padding, mode="constant")
                else:
                    # Truncate if sequence is too long
                    sequence = sequence[: self.pad_to]

                data_list.append(sequence)

            # Stack all sequences into a single array
            data_array = np.stack(data_list, axis=0)

            # Save processed data
            output_path = self.processed_folder / f"{split}_data.npy"
            np.save(output_path, data_array)
            logger.info(f"Saved {split} data: {len(data_list)} samples -> {output_path}")

        logger.info("Preprocessing complete!")

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Load data and labels for the current stage.

        Returns:
            Tuple of (data, labels) where:
                - data: np.ndarray of shape (n_samples, pad_to, 4)
                - labels: np.ndarray of shape (n_samples,) with integer labels 0-9

        Raises:
            ValueError: If stage is invalid.
        """
        if self.stage == "test":
            # Load test set
            data = np.load(self.processed_folder / "test_data.npy")
            labels = np.loadtxt(self.raw_folder / "testlabels.txt", dtype=int)
            return data, labels

        # Load training set
        data = np.load(self.processed_folder / "train_data.npy")
        labels = np.loadtxt(self.raw_folder / "trainlabels.txt", dtype=int)

        # Split into train/val based on train_val_split ratio
        split_idx = int(len(labels) * self.train_val_split)

        if self.stage == "train":
            return data[:split_idx], labels[:split_idx]
        elif self.stage == "val":
            return data[split_idx:], labels[split_idx:]
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    @staticmethod
    def convert_to_absolute_coordinates(
        sequence: np.ndarray,
    ) -> np.ndarray:
        """Convert relative coordinates (dx, dy) to absolute coordinates (x, y).

        The raw data contains pen movements (dx, dy) relative to the previous point.
        This method converts them to absolute (x, y) coordinates starting from (0, 0).

        Args:
            sequence: Array of shape (seq_len, 4) with columns [dx, dy, stroke_end, digit_end].

        Returns:
            Array of shape (seq_len, 4) with columns [x, y, stroke_end, digit_end].

        Example:
            >>> seq = np.array([[1, 2, 0, 0], [3, 1, 0, 0]])  # dx, dy movements
            >>> abs_seq = MNISTSeq.convert_to_absolute_coordinates(seq)
            >>> # abs_seq = [[1, 2, 0, 0], [4, 3, 0, 0]]  # cumulative x, y positions
        """
        # Cumulative sum converts dx -> x, dy -> y
        x = np.cumsum(sequence[:, 0])
        y = np.cumsum(sequence[:, 1])

        # Keep stroke markers unchanged
        stroke_end = sequence[:, 2]
        digit_end = sequence[:, 3]

        return np.stack([x, y, stroke_end, digit_end], axis=1)

    @staticmethod
    def plot_batch(
        sequences: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        num_samples: int = 16,
        save_path: Path | str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """Plot samples from a batch by visualizing sequences as drawn digits.

        Converts sequences back to images by drawing strokes on a 28x28 canvas.

        Args:
            sequences: Batch of sequences, shape (batch_size, pad_to, 4).
            labels: Ground truth labels, shape (batch_size,) with integer labels.
            num_samples: Number of samples to plot (default 16 for 4x4 grid).
            save_path: If provided, saves figure to this path. Otherwise, displays it.
            figsize: Figure size as (width, height). If None, defaults to (12, 12).

        Example:
            >>> dataset = MNISTSeq(root="./data", train=True)
            >>> # Plot first 16 samples
            >>> seqs = np.stack([dataset[i][0].numpy() for i in range(16)])
            >>> labs = np.array([dataset[i][1].numpy() for i in range(16)])
            >>> dataset.plot_batch(seqs, labs)
        """
        # Convert tensors to numpy if needed
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        # Limit to requested number of samples
        num_samples = min(num_samples, len(sequences))

        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_samples)))
        if figsize is None:
            figsize = (12, 12)
        fig, axs = plt.subplots(grid_size, grid_size, figsize=figsize)
        axs = axs.flatten()

        for i in range(num_samples):
            # Convert sequence to absolute coordinates
            abs_coords = MNISTSeq.convert_to_absolute_coordinates(sequences[i])

            # Create empty canvas (29x29 to avoid index errors)
            canvas = np.zeros((29, 29))

            # Draw strokes on canvas by iterating through coordinates
            for x, y in zip(abs_coords[:, 0], abs_coords[:, 1]):
                # Clip to canvas bounds
                x_int = int(np.clip(x, 0, 28))
                y_int = int(np.clip(y, 0, 28))
                canvas[y_int, x_int] = 1

            # Display the canvas
            axs[i].imshow(canvas, cmap="gray")
            axs[i].axis("off")

            # Create title with label
            label_idx = int(labels[i])
            axs[i].set_title(f"Label: {label_idx}", fontsize=10)

        # Hide unused subplots
        for i in range(num_samples, len(axs)):
            axs[i].axis("off")

        plt.tight_layout()

        # Save or show
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_samples(
        self,
        num_samples: int = 16,
        start_idx: int = 0,
        save_path: Path | str | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> None:
        """Plot samples from this dataset by visualizing sequences as drawn digits.

        This is a convenient instance method that directly uses the dataset's data.

        Args:
            num_samples: Number of samples to plot (default 16 for 4x4 grid).
            start_idx: Index to start from (default 0).
            save_path: If provided, saves figure to this path. Otherwise, displays it.
            figsize: Figure size as (width, height). If None, defaults to (12, 12).

        Example:
            >>> dataset = MNISTSeq(root="./data", train=True, download=True)
            >>> dataset.plot_samples(num_samples=16)  # Plot first 16 samples
            >>> # Plot 9 samples starting from index 100
            >>> dataset.plot_samples(num_samples=9, start_idx=100)
        """
        # Limit to available data
        num_samples = min(num_samples, len(self) - start_idx)
        if num_samples <= 0:
            logger.warning(f"No samples to plot (start_idx={start_idx}, dataset_size={len(self)})")
            return

        # Get sequences and labels for the requested samples
        sequences = self.data[start_idx : start_idx + num_samples]
        labels = self.labels[start_idx : start_idx + num_samples]

        # Use the static method to do the actual plotting
        self.plot_batch(
            sequences, labels, num_samples=num_samples, save_path=save_path, figsize=figsize
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (data, label) where:
                - data: Tensor of shape (pad_to, 4) with sequence features
                - label: Integer tensor with digit label (0-9)
        """
        # Convert to PyTorch tensors
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        # Apply optional transforms
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y

    def __len__(self) -> int:
        """Return the total number of samples in this split.

        Returns:
            Number of samples.
        """
        return len(self.labels)

    def extra_repr(self) -> str:
        """String representation for dataset printing."""
        return f"Split: {self.stage.capitalize()}"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.ion()  # Interactive mode - non-blocking plots

    # Create dataset in examples/data/ directory
    data_root = Path(__file__).parent / "data_dir"

    train_data = MNISTSeq(root=data_root, train=True, download=True)
    print(f"Train data shape: {train_data.data.shape}")
    print(f"Train labels shape: {train_data.labels.shape}")
    train_data.plot_samples(num_samples=16)  # Much cleaner!

    test_data = MNISTSeq(root=data_root, train=False, download=True)
    print(f"Test data shape: {test_data.data.shape}")
    print(f"Test labels shape: {test_data.labels.shape}")
    test_data.plot_samples(num_samples=16)  # Much cleaner!

    # Keep windows open until user presses Enter
    input("\nPress Enter to close all windows...")
    plt.close("all")
