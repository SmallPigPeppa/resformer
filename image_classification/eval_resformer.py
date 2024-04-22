import os
from typing import Callable, Optional, Sequence, Union
import pandas as pd
import pytorch_lightning as pl
import timm.models
from pytorch_lightning.cli import LightningArgumentParser
from torch.nn import CrossEntropyLoss
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.models import create_model
import models
import torch


class ClassificationEvaluator(pl.LightningModule):
    def __init__(
            self,
            weights: str,
            num_classes: int,
            image_size: int = 224,
            patch_size: int = 16,
            resize_type: str = "pi",
            results_path: Optional[str] = None, ):
        """Classification Evaluator

        Args:
            weights: Name of model weights
            n_classes: Number of target class.
            image_size: Size of input images
            patch_size: Resized patch size
            resize_type: Patch embed resize method. One of ["pi", "interpolate"]
            results_path: Path to write evaluation results. Does not write results if empty
        """
        super().__init__()
        self.save_hyperparameters()
        self.weights = weights
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.resize_type = resize_type
        self.results_path = results_path

        # Load original weights
        print(f"Loading weights {self.weights}")
        model = create_model(
            'resformer_base_patch16',
            img_size=[224],
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.2,
            drop_block_rate=None,
            use_checkpoint=False,
        )
        checkpoint_path = 'resformer_base_patch16_mr_128_160_224.pth'
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
        self.net = model
        self.net.load_state_dict(state_dict, strict=True)

        # Define metrics
        self.acc = Accuracy(num_classes=self.num_classes, task="multiclass", top_k=1)

        # Define loss
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def test_step(self, batch, _):
        x, y = batch
        x = F.interpolate(x, size=self.image_size, mode='bilinear')

        # Pass through network
        pred = self(x)
        loss = self.loss_fn(pred, y)

        # Get accuracy
        acc = self.acc(pred, y)

        # Log
        self.log_dict({'test_loss': loss, 'test_acc': acc}, sync_dist=True, on_epoch=True)

        return loss

    def test_epoch_end(self, outputs):
        if self.results_path:
            acc = self.acc.compute().detach().cpu().item()
            acc = acc * 100
            if self.trainer.is_global_zero:
                column_name = f"{self.image_size}_{self.patch_size}"

                if os.path.exists(self.results_path):

                    results_df = pd.read_csv(self.results_path, index_col=0)

                    results_df[column_name] = acc
                else:

                    results_df = pd.DataFrame({column_name: [acc]})

                    os.makedirs(os.path.dirname(self.results_path), exist_ok=True)

                results_df.to_csv(self.results_path)


if __name__ == "__main__":
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
    parser.add_lightning_class_args(ClassificationEvaluator, "model")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--works", type=int, default=4)
    parser.add_argument("--root", type=str, default='./data')
    args = parser.parse_args()
    args["logger"] = False  # Disable saving logging artifacts
    trainer = pl.Trainer.from_argparse_args(args)
    for image_size, patch_size in [(28, 2), (42, 3), (56, 4), (70, 5), (84, 6), (98, 7), (112, 8), (126, 9), (140, 10),
                                   (154, 11), (168, 12),
                                   (182, 13), (196, 14), (210, 15), (224, 16), (238, 17), (252, 18)]:
        args["model"].image_size = image_size
        args["model"].patch_size = patch_size
        model = ClassificationEvaluator(**args["model"])
        data_config = timm.data.resolve_model_data_config(model.net)
        transform = timm.data.create_transform(**data_config, is_training=False)
        val_dataset = ImageFolder(root=os.path.join(args.root, 'val'), transform=transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.works,
                                shuffle=False, pin_memory=True)
        trainer.test(model, dataloaders=val_loader)
