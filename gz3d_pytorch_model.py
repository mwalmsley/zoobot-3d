#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
from torch.nn import functional as F
import wandb
import torchvision
import pytorch_lightning as pl
# from torchmetrics import Accuracy

# re-using from Zoobot
from zoobot.pytorch.estimators import define_model, efficientnet_custom
from zoobot.pytorch.estimators.custom_layers import PermaDropout

# new custom layers
from custom_layers import DownSample, UpSample, ConvBlock, ResNet

class ZooBot3D(define_model.GenericLightningModule):
    def __init__(self,
                 output_dim = 34,
                 input_size = 128,
                 n_channels=3,
                 n_filters=32,
                 dim_mults=(1, 2, 4, 8),
                 n_classes=4,  # sets output dim 1 
                 drop_rates=(0,0,0.3,0.3),
                 test_time_dropout=False,
                 head_dropout=0.5,
                 question_index_groups=None,
                 learning_rate=1e-3,
                 weight_decay=0.05,
                 use_vote_loss=True,
                 use_seg_loss=True,
                 seg_loss_weighting=1.
                 ):
        super().__init__()
        self.channels = n_channels
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.use_vote_loss = use_vote_loss
        self.use_seg_loss = use_seg_loss
        self.seg_loss_weighting = seg_loss_weighting

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
        # TODO Not really sure how Torch parses the loss functions yet, need to handle the multiple outputs
        # Mike: as long as make step returns a dict with 'loss' key, it should work automatically
        self.dirichlet_loss = define_model.get_dirichlet_loss_func(question_index_groups)
        self.seg_loss = F.mse_loss

        # build decoder
        self.decoder = pytorch_decoder_module(self.in_out,
                                              n_classes)
                     
    def forward(self, x):

        x, h = self.encoder(x)

        z = self.head(x)

        y = self.decoder((x, h))

        return z, y
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )  

    def make_step(self, batch, batch_idx, step_name):
        ### This is going to override the generic make step, needs to get predictions for both labels and seg maps
        ### Possibly where we skip sep maps for images that don't need them?
        # x, (labels, seg_maps) = batch

        # called with train, validation, test. Not called with predict (not a step)

        pred_labels, pred_maps = self(batch['image'])  # forward pass of both encoder and decoder
        loss = self.calculate_and_log_loss((pred_labels, pred_maps), batch, step_name)      
        outputs = {
            'predictions': pred_labels,
            'predicted_maps': pred_maps,
            'loss': loss
        }
        outputs.update(batch)  # include the inputs as well, for convenience
        return outputs

    def calculate_and_log_loss(self, predictions, batch, step_name):
        # loss logging, found it!
        pred_labels, pred_maps = predictions
        loss = 0.

        if self.use_vote_loss:
            # self.loss_func returns shape of (galaxy, question), mean to ()
            multiq_loss = self.dirichlet_loss(pred_labels, batch['label_cols'], sum_over_questions=False)
            multiq_loss_reduced = torch.mean(multiq_loss)
            # TODO Dirichlet loss is trivially 0 for galaxies with no labels, which may add weighting problems
            loss += multiq_loss_reduced

            # optional extra logging
            self.log(f'{step_name}/epoch_vote_loss:0', multiq_loss_reduced, on_epoch=True, on_step=False, sync_dist=True)
            # self.log_loss_per_question(multiq_loss, prefix=step_name)

        if self.use_seg_loss:
            # we will always have 'spiral' and 'bar_mask' batch keys, else batch elements wouldn't be stackable
            seg_maps = torch.concat([batch['spiral_mask'], batch['bar_mask']], dim=1)
            # each seg map, if in the batch dict, is called e.g. spiral_mask, bar_mask, etc
            # masks are input as (batch, 1, 128, 128), where 1st dim is (dummy) channel
            # concat as (batch, map_index, 128, 128), where 1st dim is now map index
            seg_loss = self.seg_loss(seg_maps, pred_maps, reduction='none')  # shape (batch, map_index, 128, 128)]
            # set nan where seg map max is 0 i.e. no seg labels
            # missing_maps is shape (batch, 2). True where segmap missing
            missing_maps = torch.amax(seg_maps, dim=(2, 3)) == 0
            seg_loss[missing_maps] = torch.nan
            seg_loss_reduced = torch.nanmean(seg_loss)
            seg_loss_reduced = torch.nan_to_num(seg_loss_reduced)  # will still be nan if NO masks at all in batch
            loss += self.seg_loss_weighting * seg_loss_reduced

            # optional extra logging (okay the first one is v. handy for early stopping, not optional really)
            self.log(f'{step_name}/epoch_seg_loss:0', seg_loss_reduced, on_epoch=True, on_step=False, sync_dist=True)
            self.log_loss_per_seg_map_name(seg_loss, prefix=step_name)

        self.log(f'{step_name}/epoch_total_loss:0', loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss
        
    def log_loss_per_question(self, multiq_loss, prefix):
        for question_n in range(multiq_loss.shape[1]):
            self.log(f'{prefix}/epoch_questions/question_{question_n}_loss:0', torch.nanmean(multiq_loss[:, question_n]), on_epoch=True, on_step=False, sync_dist=True)
            
    def log_loss_per_seg_map_name(self, seg_loss, prefix):
        # log seg maps individually
        for seg_map_class_index in range(seg_loss.shape[1]):  # first dim is the seg map index
            # mean over the batch and all pixels, for the current seg map index
            self.log(f'{prefix}/epoch_seg_maps/seg_map_{seg_map_class_index}_loss:0', torch.nanmean(seg_loss[:, seg_map_class_index, :, :]), on_epoch=True, on_step=False, sync_dist=True)

    def log_outputs(self, outputs, step_name):
        max_images = 5
        for mask_name, mask_index in [('spiral', 0), ('bar', 1)]:

            # B1HW shape
            has_mask = torch.amax(outputs[f'{mask_name}_mask'], dim=(1, 2, 3)) > 0
            if torch.sum(has_mask) == 0:
                continue  # on to the next, don't bother with this one

            galaxy_image = wandb.Image(
                torchvision.utils.make_grid(outputs['image'][has_mask][:max_images])
            )   
            # https://docs.wandb.ai/guides/integrations/lightning#log-images-text-and-more
            # https://github.com/Lightning-AI/lightning/discussions/6723 
            self.trainer.logger.experiment.log(
                {f"{step_name}/{mask_name}_galaxy_image": galaxy_image},
                step=self.global_step
            )

            predicted_mask_image = wandb.Image(
                torchvision.utils.make_grid(outputs['predicted_maps'][has_mask][:max_images, mask_index:mask_index+1]),
            )    
            self.trainer.logger.experiment.log(
                {f"{step_name}/{mask_name}_mask_predicted": predicted_mask_image},
                step=self.global_step
            )

            true_mask_image = wandb.Image(
                torchvision.utils.make_grid(outputs[f'{mask_name}_mask'][has_mask][:max_images]),
            )    
            self.trainer.logger.experiment.log(
                {f"{step_name}/{mask_name}_mask_true": true_mask_image},
                step=self.global_step
            )


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

            self.downs.append(torch.nn.ModuleList([
                ResNet(dim_in, dim_out),
                ResNet(dim_out, dim_out),
                nn.Dropout(drop_rates[ind]) if drop_rates[ind] > 0 else nn.Identity(),
                DownSample(dim_out, dim_out) if not is_last else nn.Identity()
            ]))

        self.mid_block1 = ResNet(dim_out, dim_out)
        self.mid_block2 = ResNet(dim_out, dim_out)

        self.downs = torch.nn.ModuleList(self.downs)

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
        dropout_layer = PermaDropout
    else:
        logging.info('Not using test-time dropout')
        dropout_layer = torch.nn.Dropout
    modules_to_use.append(dropout_layer(dropout_rate))
    # TODO could optionally add a bottleneck layer here
    modules_to_use.append(efficientnet_custom.custom_top_dirichlet(encoder_dim, output_dim))

    return nn.Sequential(*modules_to_use)


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

            self.ups.append(torch.nn.ModuleList(
                [
                    ResNet(dim_out*2, dim_in),
                    ResNet(dim_in, dim_in),
                    UpSample(dim_in, dim_in) if not is_last else nn.Identity()
                ])
            )


        self.final_conv = nn.Sequential(ConvBlock(in_out[0][1], in_out[0][0]),
            nn.Mish(),
            nn.Conv2d(in_out[0][0], n_classes, 1, padding='same'),
            nn.ReLU(),
        )

        self.ups = nn.ModuleList(self.ups)


    def forward(self, inputs):

        x, h = inputs # plit the encoder output and skip connections

        for rn1, rn2, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = rn1(x)
            x = rn2(x)
            x = up(x)

        return self.final_conv(x)


if __name__ == '__main__':

    model = ZooBot3D()
    encoder = model.encoder
    decoder = model.decoder

    # test with dummy inputs
    image = torch.rand((1, 3, 128, 128))
    encoded_image, skip_outputs = encoder(image)
    # print(image.shape)
    print(encoded_image.shape)
    decoded_output = decoder((encoded_image, skip_outputs))
    print(decoded_output.shape)

    # see train.py for use
