import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch


class AnchorHeadMultiScale(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

#        self.conv_cls = nn.Conv2d(
#            input_channels, self.num_anchors_per_location * self.num_class,
#            kernel_size=1
#        )
#        self.conv_box = nn.Conv2d(
#            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
#            kernel_size=1
#        )

#        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
#            self.conv_dir_cls = nn.Conv2d(
#                input_channels,
#                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
#                kernel_size=1
#            )
#        else:
#            self.conv_dir_cls = None

        ## For car:
        self.conv_cls_car = nn.Conv2d(
                input_channels, self.num_anchors_per_location,
                kernel_size=1
            )
        self.conv_box_car = nn.Conv2d(
                input_channels, 2 * self.box_coder.code_size,
                kernel_size=1
            )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls_car = nn.Conv2d(
                input_channels,
                2 * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls_car = None

        ## For pedestrian:
        self.conv_cls_ped = nn.Conv2d(
                input_channels//2, self.num_anchors_per_location,
                kernel_size=1
            )
        self.conv_box_ped = nn.Conv2d(
                input_channels//2, 2 * self.box_coder.code_size,
                kernel_size=1
            )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls_ped = nn.Conv2d(
                input_channels//2,
                2 * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls_ped = None

        ## For cyclist:
        self.conv_cls_cycl = nn.Conv2d(
                input_channels//2, self.num_anchors_per_location,
                kernel_size=1
            )
        self.conv_box_cycl = nn.Conv2d(
                input_channels//2, 2 * self.box_coder.code_size,
                kernel_size=1
            )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls_cycl = nn.Conv2d(
                input_channels//2,
                2 * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls_cycl = None

        self.init_weights()

    def init_weights(self):
#        pi = 0.01
#        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
#        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

        pi = 0.01
        nn.init.constant_(self.conv_cls_car.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box_car.weight, mean=0, std=0.001)

        nn.init.constant_(self.conv_cls_ped.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box_ped.weight, mean=0, std=0.001)

        nn.init.constant_(self.conv_cls_cycl.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box_cycl.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        spatial_features = data_dict['spatial_features']

        cls_preds_car = self.conv_cls_car(spatial_features_2d)
        box_preds_car = self.conv_box_car(spatial_features_2d)

        cls_preds_ped = self.conv_cls_ped(spatial_features)
        box_preds_ped = self.conv_box_ped(spatial_features)

        cls_preds_cycl = self.conv_cls_cycl(spatial_features)
        box_preds_cycl = self.conv_box_cycl(spatial_features)

        if self.conv_dir_cls_car is not None:
            dir_cls_preds_car = self.conv_dir_cls_car(spatial_features_2d)
            dir_cls_preds_ped = self.conv_dir_cls_ped(spatial_features)
            dir_cls_preds_cycl = self.conv_dir_cls_cycl(spatial_features)

            dir_cls_preds_cat = torch.cat((dir_cls_preds_car, dir_cls_preds_ped, dir_cls_preds_cycl), 1)

#            print ("dir_cls_preds_cat.shape concatenated 1 is:", dir_cls_preds_cat.shape)
#            print ("\n")
            dir_cls_preds_cat = dir_cls_preds_cat.permute(0, 2, 3, 1).contiguous()
#            print ("dir_cls_preds_cat.shape concatenated 2 is:", dir_cls_preds_cat.shape)
#            print ("\n")

            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds_cat
        else:
            dir_cls_preds_cat = None

#        print ("cls_preds_car.shape 1 is:", cls_preds_car.shape)
#        print ("cls_preds_ped.shape 1 is:", cls_preds_ped.shape)
#        print ("cls_preds_cycl.shape 1 is:", cls_preds_cycl.shape)
#        print ("\n")

        cls_preds_cat = torch.cat((cls_preds_car, cls_preds_ped, cls_preds_cycl), 1)
        box_preds_cat = torch.cat((box_preds_car, box_preds_ped, box_preds_cycl), 1)

#        print ("cls_preds_cat.shape concatenated 1 is:", cls_preds_cat.shape)
#        print ("box_preds_cat.shape concatenated 1 is:", box_preds_cat.shape)
#        print ("\n")

        cls_preds_cat = cls_preds_cat.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds_cat = box_preds_cat.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

#        print ("cls_preds_cat.shape concatenated 2 is:", cls_preds_cat.shape)
#        print ("box_preds_cat.shape concatenated 2 is:", box_preds_cat.shape)
#        print ("\n")

        self.forward_ret_dict['cls_preds'] = cls_preds_cat
        self.forward_ret_dict['box_preds'] = box_preds_cat

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds_cat, box_preds=box_preds_cat, dir_cls_preds=dir_cls_preds_cat
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False







#        cls_preds = self.conv_cls(spatial_features_2d)
#        box_preds = self.conv_box(spatial_features_2d)

#        print ("spatial_features_2d.shape is:", spatial_features_2d.shape)
#        print ("spatial_features.shape is:", spatial_features.shape)
#        print ("\n")

#        print ("cls_preds.shape 1 is:", cls_preds.shape)
#        print ("box_preds.shape 1 is:", box_preds.shape)
#        print ("\n")




#        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
#        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

#        print ("cls_preds.shape 2 is:", cls_preds.shape)
#        print ("box_preds.shape 2 is:", box_preds.shape)
#        print ("\n")

#        self.forward_ret_dict['cls_preds'] = cls_preds
#        self.forward_ret_dict['box_preds'] = box_preds

#        if self.conv_dir_cls is not None:
#            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
#            print ("dir_cls_preds.shape 1 is:", dir_cls_preds.shape)
#            print ("\n")
#            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#            print ("dir_cls_preds.shape 2 is:", dir_cls_preds.shape)
#            print ("\n")

#            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
#        else:
#            dir_cls_preds = None

#        if self.training:
#            targets_dict = self.assign_targets(
#                gt_boxes=data_dict['gt_boxes']
#            )
#            self.forward_ret_dict.update(targets_dict)

#        if not self.training or self.predict_boxes_when_training:
#            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
#                batch_size=data_dict['batch_size'],
#                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
#            )
#            data_dict['batch_cls_preds'] = batch_cls_preds
#            data_dict['batch_box_preds'] = batch_box_preds
#            data_dict['cls_preds_normalized'] = False

#        exit()
        return data_dict
