#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy

import pytorch_losses as losses
from custom_layers import DownSample, UpSample, ConvBlock, ResNet
import efficientnet_custom

class GenericLightningModule(pl.LightningModule):
    """
    All Zoobot models use the lightningmodule API and so share this structure
    super generic, just to outline the structure. nothing specific to dirichlet, gz, etc
    only assumes an encoder and a head
    """

    def __init__(
        self,
        *args,  # to be saved as hparams
        ):
        super().__init__()
        self.save_hyperparameters()  # saves all args by default
        self.setup_metrics()


    def setup_metrics(self):
        # these are ignored unless output dim = 2
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        # self.log_on_step = False
        # self.log_on_step is useful for debugging, but slower - best when log_every_n_steps is fairly large


    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
    
    def make_step(self, batch, batch_idx, step_name):
        x, labels = batch
        predictions = self(x)  # by default, these are Dirichlet concentrations
        loss = self.calculate_and_log_loss(predictions, labels, step_name)      
        return {'loss': loss, 'predictions': predictions, 'labels': labels}

    def calculate_and_log_loss(self, predictions, labels, step_name):
        raise NotImplementedError('Must be subclassed')

    def configure_optimizers(self):
        raise NotImplementedError('Must be subclassed')

    def training_step(self, batch, batch_idx):
        return self.make_step(batch, batch_idx, step_name='train')

    def on_training_batch_end(self, outputs, *args):
        self.log_outputs(outputs, step_name='train')

    def validation_step(self, batch, batch_idx):
        return self.make_step(batch, batch_idx, step_name='validation')

    def on_validation_batch_end(self, outputs, *args):
        self.log_outputs(outputs, step_name='validation')

    def test_step(self, batch, batch_idx):
        return self.make_step(batch, batch_idx, step_name='test')

    def on_test_batch_end(self, outputs, *args):
         self.log_outputs(outputs, step_name='test')

    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#inference
        # this calls forward, while avoiding the need for e.g. model.eval(), torch.no_grad()
        # x, y = batch  # would be usual format, but here, batch does not include labels
        return self(batch)

class ZooBot3D(GenericLightningModule):
    def __init__(self,
                 output_dim = 34,
                 input_size = 128,
                 n_channels=3,
                 n_filters=32,
                 dim_mults=(1, 2, 4, 8),
                 n_classes=4,
                 drop_rates=(0,0,0.3,0.3),
                 test_time_dropout=False,
                 head_dropout=0.5,
                 question_index_groups=None
                 ):
        super().__init__()
        self.channels = n_channels
        self.input_size = input_size
        dims = [self.channels, *map(lambda m: n_filters * m, dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.drop_rates = drop_rates

        # build encoder model
        self.encoder = pytorch_encoder_module(self.in_out,
                                              self.drop_rates)

        ## build classifier
        self.encoder_dim = get_encoder_dim(self.encoder, self.input_size, self.channels)
        self.head = get_pytorch_dirichlet_head(self.encoder_dim,
                                               output_dim,
                                               test_time_dropout,
                                               head_dropout)
        # Losses - Set both separately then just add together? MSE for real seg maps, BXE for threshold
        # Not really sure how Torch parses the loss functions yet, need to handle the multiple outputs
        self.dirichlet_loss = get_dirichlet_loss_func(question_index_groups)
        self.seg_loss = F.mse_loss

        # build decoder
        self.decoder = pytorch_decoder_module(self.in_out,
                                              n_classes)
                     
    def forward(self, x):

        x, h = self.encoder(x)

        z = self.head(x)

        y = self.decoder((x, h))

        return z, y

    def make_step(self, batch, batch_idx, step_name):
        ### This is going to override the generic make step, needs to get predictions for both labels and seg maps
        ### Possibly where we skip sep maps for images that don't need them?
        x, (labels, seg_maps) = batch
        pred_labels, pred_maps = self(x)
        loss = self.calculate_and_log_loss((pred_labels, pred_maps), (labels, seg_maps), step_name)      
        return {'loss': loss, 'predictions': pred_labels, 'labels': labels, 'predicted_maps': pred_maps, 'seg_maps': seg_maps}

    def calculate_and_log_loss(self, predictions, labels_maps, step_name):
        # loss logging, found it!
        pred_labels, pred_maps = predictions
        labels, seg_maps = labels_maps
        # self.loss_func returns shape of (galaxy, question), mean to ()
        multiq_loss = self.dirichlet_loss(pred_labels, labels, sum_over_questions=False)
        seg_loss = self.seg_loss(seg_maps, pred_maps, reduction='none') # Reduction here?
        # if hasattr(self, 'schema'):
        self.log_loss_per_question(multiq_loss, prefix=step_name)
        self.log_loss_per_seg_map(seg_loss, prefix=step_name)
        # sum over questions and take a per-device mean
        # for DDP strategy, batch size is constant (batches are not divided, data pool is divided)
        # so this will be the global per-example mean
        loss = torch.mean(torch.sum(multiq_loss, axis=1) + torch.sum(seg_loss, axis=(1,2,3)))
        return loss
        
    def log_loss_per_question(self, multiq_loss, prefix):
        # log questions individually
        # TODO need schema attribute or similar to have access to question names, this will do for now
        # unlike Finetuneable..., does not use TorchMetrics, simply logs directly
        # TODO could use TorchMetrics and for q in schema, self.q_metric loop
        for question_n in range(multiq_loss.shape[1]):
            self.log(f'{prefix}/epoch_questions/question_{question_n}_loss:0', torch.mean(multiq_loss[:, question_n]), on_epoch=True, on_step=False, sync_dist=True)
            
    def log_loss_per_seg_map(self, seg_loss, prefix):
        # log seg maps individually
        # Check syntax here, is it going to log the bar and spiral seg loss separately?
        for seg_map_n in range(seg_loss.shape[1]):
            self.log(f'{prefix}/epoch_seg_maps/seg_map_{seg_map_n}_loss:0', torch.mean(seg_loss[:, seg_map_n, :, :]), on_epoch=True, on_step=False, sync_dist=True)


# Standalone encoder class: return both the encoder output and the skip connections, include midblocks
class pytorch_encoder_module(nn.Module):
    def __init__(self, 
                 in_out,
                 drop_rates,
                 ):
        super().__init__()

        self.downs = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                ResNet(dim_in, dim_out),
                ResNet(dim_out, dim_out),
                nn.Dropout(drop_rates[ind]) if drop_rates[ind] > 0 else nn.Identity(),
                DownSample(dim_out, dim_out) if not is_last else nn.Identity()
            ])

        self.mid_block1 = ResNet(dim_out, dim_out)
        self.mid_block2 = ResNet(dim_out, dim_out)

    def forward(self, x):
        h = [] # collect the skip conection outputs for the decoder

        for rn1, rn2, drop, down in self.downs:
            x = rn1(x)
            x = rn2(x)
            x = drop(x)
            h.append(x) # outputs for skip connections
            x = down(x)

        x = self.mid_block1(x)
        x = self.mid_block2(x)

        return x, h # return midblock output and skip connections

def get_encoder_dim(encoder, input_size, channels):
    x = torch.randn(1, channels, input_size, input_size)  # batch size of 1
    return encoder(x)[0].shape[1] # return number of channels, global pooling added later

# zoobot classification head with dirchelet loss

def get_pytorch_dirichlet_head(encoder_dim: int, output_dim: int, test_time_dropout: bool, dropout_rate: float) -> torch.nn.Sequential:
    """
    NOTE: From Zoobot. Change, include Pooling layer.
    Head to combine with encoder (above) when predicting Galaxy Zoo decision tree answers.
    Pytorch Sequential model.
    Predicts Dirichlet concentration parameters.
    
    Also used when finetuning on a new decision tree - see :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree`.

    Args:
        encoder_dim (int): dimensions of preceding encoder i.e. the input size expected by this submodel.
        output_dim (int): output dimensions of this head e.g. 34 to predict 34 answers.
        test_time_dropout (bool): Use dropout at test time. 
        dropout_rate (float): P of dropout. See torch.nn.Dropout docs.

    Returns:
        torch.nn.Sequential: pytorch model expecting `encoder_dim` vector and predicting `output_dim` decision tree answers.
    """

    modules_to_use = []

    assert output_dim is not None
    # yes AdaptiveAvgPool2d, encoder output needs to be for both classifier and decoder  
    pooling_layer = nn.AdaptiveAvgPool2d(1)
    modules_to_use.append(pooling_layer)
    modules_to_use.append(nn.Flatten(start_dim=1))
    if test_time_dropout:
        logging.info('Using test-time dropout')
        dropout_layer = custom_layers.PermaDropout
    else:
        logging.info('Not using test-time dropout')
        dropout_layer = torch.nn.Dropout
    modules_to_use.append(dropout_layer(dropout_rate))
    # TODO could optionally add a bottleneck layer here
    modules_to_use.append(custom_top_dirichlet(encoder_dim, output_dim))

    return nn.Sequential(*modules_to_use)

def custom_top_dirichlet(input_dim, output_dim):
    """
    Final dense layer used in GZ DECaLS (after global pooling). 
    ``output_dim`` neurons with an activation of ``tf.nn.sigmoid(x) * 100. + 1.``, chosen to ensure 1-100 output range
    This range is suitable for parameters of Dirichlet distribution.

    Args:
        output_dim (int): Dimension of dense layer e.g. 34 for decision tree with 34 answers

    Returns:
        nn.Sequential: nn.Linear followed by 1-101 sigmoid activation
    """
    return nn.Sequential(
        # LinearWithCustomInit(in_features=input_dim, out_features=output_dim),
        nn.Linear(in_features=input_dim, out_features=output_dim),
        ScaledSigmoid()
    )

class ScaledSigmoid(nn.modules.Sigmoid):
    # https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#ReLU

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): any vector. typically logits from a neural network

        Returns:
            Tensor: input mapped to range (1, 101) via torch.sigmoid
        """
        return torch.sigmoid(input) * 100. + 1.  # could make args if I needed


def get_dirichlet_loss_func(question_index_groups):
    # This just adds schema.question_index_groups as an arg to the usual (labels, preds) loss arg format
    # Would use lambda but multi-gpu doesn't support as lambda can't be pickled
    return partial(dirichlet_loss, question_index_groups=question_index_groups)


    # accept (labels, preds), return losses of shape (batch, question)
def dirichlet_loss(preds, labels, question_index_groups, sum_over_questions=False):
    # pytorch convention is preds, labels for loss func
    # my and sklearn convention is labels, preds for loss func

    # multiquestion_loss returns loss of shape (batch, question)
    # torch.sum(multiquestion_loss, axis=1) gives loss of shape (batch). Equiv. to non-log product of question likelihoods.
    multiq_loss = losses.calculate_multiquestion_loss(labels, preds, question_index_groups)
    if sum_over_questions:
        return torch.sum(multiq_loss, axis=1)
    else:
        return multiq_loss


# decoder clss, include the skip connections — how to save on GPU cycles by not forward passing unecessarily?

class pytorch_decoder_module(nn.Module):
    def __init__(self,
                 in_out,
                 n_classes,
                 ):
        super().__init__()

        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                ResNet(dim_out*2, dim_in),
                ResNet(dim_in, dim_in),
                UpSample(dim_in, dim_in) if not is_last else nn.Identity()
            ])


        self.final_conv = nn.Sequential(ConvBlock(in_out[0][1], in_out[0][0]),
            nn.Mish(),
            nn.Conv2d(in_out[0][0], n_classes, 1, padding='same'),
            nn.ReLU(),
        )


    def forward(self, inputs):

        x, h = inputs # plit the encoder output and skip connections

        for rn1, rn2, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = rn1(x)
            x = rn2(x)
            x = up(x)

        return self.final_conv(x)

































