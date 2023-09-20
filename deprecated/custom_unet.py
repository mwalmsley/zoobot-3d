# https://rising.readthedocs.io/en/stable/lightning_segmentation.html
import pytorch_lightning as pl
import torch

from typing import Optional


class Unet(pl.LightningModule):
    """Simple U-Net without training logic"""
    def __init__(self, hparams: dict):
        """
        Args:
            hparams: the hyperparameters needed to construct the network.
                Specifically these are:
                * start_filts (int)
                * depth (int)
                * in_channels (int)
                * num_classes (int)
        """
        super().__init__()
        # 4 downsample layers
        out_filts = hparams.get('start_filts', 16)
        depth = hparams.get('depth', 3)
        in_filts = hparams.get('in_channels', 1)
        num_classes = hparams.get('num_classes', 2)

        for idx in range(depth):
            down_block = torch.nn.Sequential(
                torch.nn.Conv3d(in_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True)
            )
            in_filts = out_filts
            out_filts *= 2

            setattr(self, 'down_block_%d' % idx, down_block)

        out_filts = out_filts // 2
        in_filts = in_filts // 2
        out_filts, in_filts = in_filts, out_filts

        for idx in range(depth-1):
            up_block = torch.nn.Sequential(
                torch.nn.Conv3d(in_filts + out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(out_filts, out_filts, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True)
            )

            in_filts = out_filts
            out_filts = out_filts // 2

            setattr(self, 'up_block_%d' % idx, up_block)

        self.final_conv = torch.nn.Conv3d(in_filts, num_classes, kernel_size=1)
        self.max_pool = torch.nn.MaxPool3d(2, stride=2)
        self.up_sample = torch.nn.Upsample(scale_factor=2)
        self.hparams = hparams

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forwards the :attr`input_tensor` through the network to obtain a prediction

        Args:
            input_tensor: the network's input

        Returns:
            torch.Tensor: the networks output given the :attr`input_tensor`
        """
        depth = self.hparams.get('depth', 3)

        intermediate_outputs = []

        # Compute all the encoder blocks' outputs
        for idx in range(depth):
            intermed = getattr(self, 'down_block_%d' % idx)(input_tensor)
            if idx < depth - 1:
                # store intermediate values for usage in decoder
                intermediate_outputs.append(intermed)
                input_tensor = getattr(self, 'max_pool')(intermed)
            else:
                input_tensor = intermed

        # Compute all the decoder blocks' outputs
        for idx in range(depth-1):
            input_tensor = getattr(self, 'up_sample')(input_tensor)

            # use intermediate values from encoder
            from_down = intermediate_outputs.pop(-1)
            intermed = torch.cat([input_tensor, from_down], dim=1)
            input_tensor = getattr(self, 'up_block_%d' % idx)(intermed)

        return getattr(self, 'final_conv')(input_tensor)
    

class TrainableUNet(Unet):
    """A trainable UNet (extends the base class by training logic)"""
    def __init__(self, hparams: Optional[dict] = None):
        """
        Args:
            hparams: the hyperparameters needed to construct and train the network.
                Specifically these are:
                * start_filts (int)
                * depth (int)
                * in_channels (int)
                * num_classes (int)
                * min_scale (float)
                * max_scale (float)
                * min_rotation (int, float)
                * max_rotation (int, float)
                * batch_size (int)
                * num_workers(int)
                * learning_rate (float)

                For all of them exist usable default parameters.
        """
        if hparams is None:
            hparams = {}
        super().__init__(hparams)

        # define loss functions
        # TODO change these
        # self.dice_loss = SoftDiceLoss(weight=[0., 1.])
        # self.ce_loss = torch.nn.CrossEntropyLoss()


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimier to use for training

        Returns:
            torch.optim.Optimier: the optimizer for updating the model's parameters
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get('learning_rate', 1e-3))

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Defines the training logic

        Args:
            batch: contains the data (inputs and ground truth)
            batch_idx: the number of the current batch

        Returns:
            dict: the current loss value
        """
        x, y = batch['data'], batch['label']

        # remove channel dim from gt (was necessary for augmentation)
        y = y[:, 0].long()

        # obtain predictions
        pred = self(x)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)

        # Calculate losses
        ce_loss = self.ce_loss(pred, y)
        dice_loss = self.dice_loss(softmaxed_pred, y)
        total_loss = (ce_loss + dice_loss) / 2

        # calculate dice coefficient
        # dice_coeff = binary_dice_coefficient(torch.argmax(softmaxed_pred, dim=1), y)

        # log values
        # self.logger.experiment.add_scalar('Train/DiceCoeff', dice_coeff)
        self.logger.experiment.add_scalar('Train/CE', ce_loss)
        self.logger.experiment.add_scalar('Train/SoftDiceLoss', dice_loss)
        self.logger.experiment.add_scalar('Train/TotalLoss', total_loss)

        return {'loss': total_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Defines the validation logic

        Args:
            batch: contains the data (inputs and ground truth)
            batch_idx: the number of the current batch

        Returns:
            dict: the current loss and metric values
        """
        x, y = batch['data'], batch['label']

        # remove channel dim from gt (was necessary for augmentation)
        y = y[:, 0].long()

        # obtain predictions
        pred = self(x)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)

        # calculate losses
        ce_loss = self.ce_loss(pred, y)
        dice_loss = self.dice_loss(softmaxed_pred, y)
        total_loss = (ce_loss + dice_loss) / 2

        # calculate dice coefficient
        # dice_coeff = binary_dice_coefficient(torch.argmax(softmaxed_pred, dim=1), y)

        # log values
        # self.logger.experiment.add_scalar('Val/DiceCoeff', dice_coeff)
        self.logger.experiment.add_scalar('Val/CE', ce_loss)
        self.logger.experiment.add_scalar('Val/SoftDiceLoss', dice_loss)
        self.logger.experiment.add_scalar('Val/TotalLoss', total_loss)

        # return {'val_loss': total_loss, 'dice': dice_coeff}

    def validation_epoch_end(self, outputs: list) -> dict:
        """Aggregates data from each validation step

        Args:
            outputs: the returned values from each validation step

        Returns:
            dict: the aggregated outputs
        """
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()

        # tqdm.write('Dice: \t%.3f' % mean_outputs['dice'].item())
        return mean_outputs