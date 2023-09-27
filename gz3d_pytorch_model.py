#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn
from torch.nn import functional as F
import wandb
import torchvision
import pytorch_lightning as pl
from pyro.distributions import BetaBinomial
# from torchmetrics import Accuracy

# re-using from Zoobot
from zoobot.pytorch.estimators import define_model, efficientnet_custom
from zoobot.pytorch.estimators.custom_layers import PermaDropout

# new custom layers
from custom_layers import DownSample, UpSample, ConvBlock, ResNet


class ZooBot3D(define_model.GenericLightningModule): 
    def __init__(self,
                #  output_dim = 34,
                 input_size=128,
                 n_channels=3,
                 n_filters=32,
                 dim_mults=[1, 2, 4, 8],
                # dim_mults=(1, 2, 4),
                 n_classes=4,  # sets output dim 1
                 drop_rates=[0, 0, 0.3, 0.3],
                # drop_rates=(0,0,0.3),
                #  test_time_dropout=False,
                #  head_dropout=0.5,
                #  question_index_groups=None,
                 learning_rate=1e-3,
                 weight_decay=0.05,
                #  use_vote_loss=True,
                #  use_seg_loss=True,
                 seg_loss_weighting=1.,
                #  vote_loss_weighting=1.,
                 seg_loss_metric='mse',
                #  skip_connection_weighting=1.,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.channels = n_channels
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # self.use_vote_loss = use_vote_loss
        # self.use_seg_loss = use_seg_loss
        self.seg_loss_weighting = seg_loss_weighting
        # self.vote_loss_weighting = vote_loss_weighting
        self.seg_loss_metric=seg_loss_metric

        # self.skip_connection_weighting = skip_connection_weighting

        dims = [self.channels, *map(lambda m: n_filters * m, dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        self.drop_rates = drop_rates

        # build encoder model
        self.encoder = pytorch_encoder_module(self.in_out,
                                              self.drop_rates)

                                            #   self.skip_connection_weighting
                                            

        ## build classifier
        self.encoder_dim = get_encoder_dim(self.encoder, self.input_size, self.channels)
        # if self.use_vote_loss:
        #     self.head = get_pytorch_dirichlet_head(self.encoder_dim,
        #                                         output_dim,
        #                                         test_time_dropout,
        #                                         head_dropout)
            # Losses - Set both separately then just add together? MSE for real seg maps, BXE for threshold
            # TODO Not really sure how Torch parses the loss functions yet, need to handle the multiple outputs
            # Mike: as long as make step returns a dict with 'loss' key, it should work automatically
            # self.dirichlet_loss = define_model.get_dirichlet_loss_func(question_index_groups)
        if self.seg_loss_metric == 'mse':
            self.seg_loss = F.mse_loss
            self.final_activation = 'relu'
        elif self.seg_loss_metric == 'l1':
            self.seg_loss = F.l1_loss
            self.final_activation = 'relu'
        elif self.seg_loss_metric == 'beta_binomial':
            self.n_classes == 4  # overrides
            self.seg_loss = beta_binomial_loss_func
            self.final_activation = 'beta_binomial'
        else:
            raise ValueError(self.seg_loss_metric)

        # build decoder
        self.decoder = pytorch_decoder_module(self.in_out,
                                              self.n_classes,
                                              final_activation=self.final_activation
                                            )
                     
    def forward(self, x):

        x, h = self.encoder(x)

        # z = self.head(x)

        y = self.decoder((x, h))

        # return z, y
        return y
    
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

        # pred_labels, pred_maps = self(batch['image'])  # forward pass of both encoder and decoder
        # loss = self.calculate_and_log_loss((pred_labels, pred_maps), batch, step_name) 

        pred_maps = self(batch['image'])
        loss = self.calculate_and_log_loss(pred_maps, batch, step_name) 
             
        outputs = {
            # 'predictions': pred_labels,
            'predicted_maps': pred_maps,
            'loss': loss
        }
        outputs.update(batch)  # include the inputs as well, for convenience
        return outputs

    def calculate_and_log_loss(self, predictions, batch, step_name):
        # pred_labels, pred_maps = predictions
        pred_maps = predictions
        
        # loss = 0

        # if self.use_vote_loss:
        #     # self.loss_func returns shape of (galaxy, question), mean to ()
        #     # dim=1 is the question dim, so sum over that
        #     has_votes = torch.amax(batch['label_cols'], dim=1) > 0
        #     if torch.any(has_votes > 0):
        #         multiq_loss = self.dirichlet_loss(pred_labels[has_votes], batch['label_cols'][has_votes], sum_over_questions=False)
        #         multiq_loss_reduced = torch.mean(multiq_loss)
        #     else:
        #         multiq_loss_reduced = 0
                
        #     # optional extra logging
        #     self.log(f'{step_name}/epoch_vote_loss:0', multiq_loss_reduced, on_epoch=True, on_step=False, sync_dist=True)
        #     # self.log_loss_per_question(multiq_loss, prefix=step_name)
            
        #     loss += (self.vote_loss_weighting * multiq_loss_reduced)


        # if self.use_seg_loss:
        # we will always have 'spiral' and 'bar_mask' batch keys, else batch elements wouldn't be stackable
        seg_maps = torch.concat([batch['spiral_mask'], batch['bar_mask']], dim=1)
        # each seg map, if in the batch dict, is called e.g. spiral_mask, bar_mask, etc
        # masks are input as (batch, 1, 128, 128), where 1st dim is (dummy) channel
        # concat as (batch, map_index, 128, 128), where 1st dim is now map index
        # set nan where seg map max is 0 i.e. no seg labels

        # has_maps is shape (batch, 2). True where spiral segmap exists
        has_maps = torch.amax(seg_maps, dim=(2, 3)) > 0  # check over the spatial dims
        # has_maps = torch.sum(seg_maps[:, 0], dim=(1, 2)) > 0
        if torch.any(has_maps > 0):
            seg_loss = self.seg_loss(seg_maps, pred_maps, reduction='none')
            seg_loss[~has_maps] = torch.nan  # set loss to 0 where no seg map exists (or could set preds to zero, or could index out)
            # seg loss has shape (batch, map_index)]
            seg_loss_reduced = torch.nanmean(seg_loss)

            # optional extra logging (okay the first one is v. handy for early stopping, not optional really)
            self.log(f'{step_name}/epoch_seg_loss:0', seg_loss_reduced, on_epoch=True, on_step=False, sync_dist=True)
            self.log_loss_per_seg_map_name(seg_loss, prefix=step_name)
        else:
            # logging.warning('No seg maps in batch, skipping seg loss')
            seg_loss_reduced = 0

            # loss += (self.seg_loss_weighting * seg_loss_reduced)
        loss = self.seg_loss_weighting * seg_loss_reduced

        self.log(f'{step_name}/epoch_total_loss:0', loss, on_epoch=True, on_step=False, sync_dist=True)

        return loss
        
    # def log_loss_per_question(self, multiq_loss, prefix):
    #     for question_n in range(multiq_loss.shape[1]):
    #         self.log(f'{prefix}/epoch_questions/question_{question_n}_loss:0', torch.mean(multiq_loss[:, question_n]), on_epoch=True, on_step=False, sync_dist=True)
            
    def log_loss_per_seg_map_name(self, seg_loss, prefix):
        # log seg maps individually
        for seg_map_class_index in range(seg_loss.shape[1]):  # first dim is the seg map index
            # mean over the batch and all pixels, for the current seg map index
            self.log(f'{prefix}/epoch_seg_maps/seg_map_{seg_map_class_index}_loss', torch.nanmean(seg_loss[:, seg_map_class_index, :, :]), on_epoch=True, on_step=False, sync_dist=True)

    def log_outputs(self, outputs, step_name):

        step = self.global_step
        if step % 100 == 0:  # for speed

            max_images = 5
            for mask_name, mask_index in [('spiral', 0), ('bar', 1)]:
            # for mask_name, mask_index in [('spiral', 0), ('bar', 2)]:

                # B1HW shape
                has_mask = torch.amax(outputs[f'{mask_name}_mask'], dim=(1, 2, 3)) > 0
                if torch.sum(has_mask) == 0:
                    continue  # on to the next, don't bother with this one
                else:
                    galaxy_image = wandb.Image(
                        torchvision.utils.make_grid(outputs['image'][has_mask][:max_images])
                    )   
                    # https://docs.wandb.ai/guides/integrations/lightning#log-images-text-and-more
                    # https://github.com/Lightning-AI/lightning/discussions/6723 
                    self.trainer.logger.experiment.log(
                        {f"{step_name}_images/{mask_name}_galaxy_image": galaxy_image},
                        step=step
                    )

                    true_mask_image = wandb.Image(
                        torchvision.utils.make_grid(outputs[f'{mask_name}_mask'][has_mask][:max_images]),
                    )    
                    self.trainer.logger.experiment.log(
                        {f"{step_name}_images/{mask_name}_mask_true": true_mask_image},
                        step=step
                    )

                    predicted_maps = outputs['predicted_maps'][has_mask][:max_images]
                    
                    if self.seg_loss_metric == 'beta_binomial':
                        mean_images = get_beta_mean(predicted_maps[:, mask_index:mask_index+1],  predicted_maps[:, mask_index+1:mask_index+2])
                        predicted_mask_image = wandb.Image(torchvision.utils.make_grid(mean_images)) 
                        self.trainer.logger.experiment.log(
                            {f"{step_name}_images/{mask_name}_mask_predicted": predicted_mask_image},
                            step=step
                        )

                        uncertainty_mask_image = wandb.Image(
                            torchvision.utils.make_grid(
                                # mean = a / (a+b)
                                get_beta_variance(predicted_maps[:, mask_index:mask_index+1], predicted_maps[:, mask_index+1:mask_index+2])
                            ),
                        )    
                        self.trainer.logger.experiment.log(
                            {f"{step_name}_images/{mask_name}_mask_uncertainty": uncertainty_mask_image},
                            step=step
                        )
                    else:
                        predicted_mask_image = wandb.Image(
                            torchvision.utils.make_grid(predicted_maps[:, mask_index:mask_index+1])
                        )    
                        self.trainer.logger.experiment.log(
                            {f"{step_name}_images/{mask_name}_mask_predicted": predicted_mask_image},
                            step=step
                        )




# class ZoobotDummy(ZooBot3D):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
        
#         self.dummy_encoder = define_model.get_pytorch_encoder(
#                 'efficientnet_b0',
#                 3,
#                 use_imagenet_weights=False
#             )
        
#     def __forward__(self, x):
#         x = self.dummy_encoder(x)
#         z = self.head(x)
#         y = torch.random(x.shape)  # 'decoded' image
#         return y, z
        

    

# Standalone encoder class: return both the encoder output and the skip connections, include midblocks
class pytorch_encoder_module(nn.Module):
    def __init__(self, 
                 in_out,
                 drop_rates
                #  skip_connection_weighting=1.
                 ):
        super().__init__()

        self.downs = []
        num_resolutions = len(in_out)

        # self.skip_connection_weighting = skip_connection_weighting

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

        # upweight skip connections coming from first encoder-block only (rn1 i.e. resnet1)
        # h = [h_i * self.skip_connection_weighting for h_i in h]
        # h[0] = h[0] * self.skip_connection_weighting

        return x, h # return midblock output and skip connections

def get_encoder_dim(encoder, input_size, channels):
    x = torch.randn(1, channels, input_size, input_size)  # batch size of 1
    return encoder(x)[0].shape[1] # return number of channels, global pooling added later

# zoobot classification head with dirchelet loss

# def get_pytorch_dirichlet_head(encoder_dim: int, output_dim: int, test_time_dropout: bool, dropout_rate: float) -> torch.nn.Sequential:
#     """
#     NOTE: From Zoobot. Change, include Pooling layer.
#     Head to combine with encoder (above) when predicting Galaxy Zoo decision tree answers.
#     Pytorch Sequential model.
#     Predicts Dirichlet concentration parameters.
    
#     Also used when finetuning on a new decision tree - see :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotTree`.

#     Args:
#         encoder_dim (int): dimensions of preceding encoder i.e. the input size expected by this submodel.
#         output_dim (int): output dimensions of this head e.g. 34 to predict 34 answers.
#         test_time_dropout (bool): Use dropout at test time. 
#         dropout_rate (float): P of dropout. See torch.nn.Dropout docs.

#     Returns:
#         torch.nn.Sequential: pytorch model expecting `encoder_dim` vector and predicting `output_dim` decision tree answers.
#     """

#     modules_to_use = []

#     assert output_dim is not None
#     # yes AdaptiveAvgPool2d, encoder output needs to be for both classifier and decoder  
#     pooling_layer = nn.AdaptiveAvgPool2d(1)
#     modules_to_use.append(pooling_layer)
#     modules_to_use.append(nn.Flatten(start_dim=1))
#     if test_time_dropout:
#         logging.info('Using test-time dropout')
#         dropout_layer = PermaDropout
#     else:
#         logging.info('Not using test-time dropout')
#         dropout_layer = torch.nn.Dropout
#     modules_to_use.append(dropout_layer(dropout_rate))
#     # TODO could optionally add a bottleneck layer here
#     modules_to_use.append(efficientnet_custom.custom_top_dirichlet(encoder_dim, output_dim))

#     return nn.Sequential(*modules_to_use)

def get_beta_mean(a, b):
    return a / (a + b)

def get_beta_variance(a, b):
    return (a * b) / ((a + b)**2 * (a + b + 1))

# decoder clss, include the skip connections — how to save on GPU cycles by not forward passing unecessarily?

class pytorch_decoder_module(nn.Module):
    def __init__(self,
                 in_out,
                 n_classes,
                 final_activation='relu'
                 ):
        super().__init__()

        self.final_activation = final_activation

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


        if self.final_activation == 'scaled_sigmoid':
            final_conv_act = efficientnet_custom.ScaledSigmoid()
        elif self.final_activation == 'relu':
            final_conv_act = nn.ReLU()
        else:
            raise ValueError(self.final_activation)
        
        self.final_conv = nn.Sequential(ConvBlock(in_out[0][1], in_out[0][0]),
            nn.Mish(),
            # one output filter per class, stride of 1
            nn.Conv2d(in_out[0][0], n_classes, 1, padding='same'),
            final_conv_act,
        )

        self.ups = nn.ModuleList(self.ups)


    def forward(self, inputs):

        x, h = inputs # split the encoder output and skip connections

        for rn1, rn2, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1) # adding skip connection as extra channel
            x = rn1(x)
            x = rn2(x)
            x = up(x)

        return self.final_conv(x)


def beta_binomial_loss_func(segmaps, pred_maps, reduction, epsilon=1e-5):
    
    # set n_classes = 4 for 4 output channels
    # assert self.n_classes == 4

    # interpret first two channels as spiral concentration1/2
    # second two as bars

    pred_maps = pred_maps + epsilon
    
    recovered_masks = (15 * segmaps).to(int)  # back to counts
    # unstack spiral/bar again
    spiral_maps, bar_maps = recovered_masks[:, 0], recovered_masks[:, 1]
    spiral_pred_c1, spiral_pred_c2 = pred_maps[:, 0], pred_maps[:, 1]
    bar_pred_c1, bar_pred_c2 = pred_maps[:, 2], pred_maps[:, 3]

    spiral_loss = BetaBinomial(spiral_pred_c1, spiral_pred_c2, total_count=15, validate_args=True).log_prob(spiral_maps)
    bar_loss = BetaBinomial(bar_pred_c1, bar_pred_c2, total_count=15, validate_args=True).log_prob(bar_maps)

    # important minus sign, maximise log prob == minimise negative log prob
    return -torch.stack([spiral_loss, bar_loss], dim=1)  # stack loss like segmaps

    # return spiral_loss + bar_loss
    


if __name__ == '__main__':

    from zoobot.shared import schemas

    batch_size = 16
    image_size = 128
    question_index_groups = [(0, 2)]
    output_dim = 3  # 0, 1, 2

    model = ZooBot3D(
        # question_index_groups=question_index_groups,
        # output_dim=output_dim
    )
    encoder = model.encoder
    decoder = model.decoder

    # test with dummy inputs
    image = torch.rand((batch_size, 3, image_size, image_size))
    encoded_image, skip_outputs = encoder(image)
    # print(image.shape)
    print(encoded_image.shape)
    decoded_output = decoder((encoded_image, skip_outputs))
    print(decoded_output.shape)

    # predicted_labels = torch.rand((batch_size, output_dim))
    predicted_maps = torch.rand((batch_size, 2, image_size, image_size))
    # predictions = predicted_labels, predicted_maps
    predictions = predicted_maps

    batch = {
        'image': image,
        # a few will have no votes, by chance
        # 'label_cols': torch.randint(low=0, high=1, size=(batch_size, output_dim)),
        'spiral_mask': torch.rand((batch_size, 1, image_size, image_size)),
        'bar_mask': torch.rand((batch_size, 1, image_size, image_size))
    }
    # randomly blank some masks
    batch['spiral_mask'][torch.rand(batch_size) < 0.5] = 0
    batch['bar_mask'][torch.rand(batch_size) < 0.5] = 0
    model.calculate_and_log_loss(predictions, batch, step_name='train')

    # see train.py for use
