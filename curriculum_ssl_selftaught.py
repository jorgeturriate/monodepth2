# curriculum_ssl_selftaught.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
import os
import networks
from layers import *

class CurriculumLearnerSelfSupervised:
    def __init__(self, opt, model, dataloader, dataset, model_path ,pacing_function="linear", device="cuda"):
        """
        model: SPIdepth model (without loading weights)
        dataloader: full training dataloader
        pacing_function: linear or quadratic pacing function
        model_path: path to the checkpoint to use for scoring
        opt: options file to load the model
        """
        self.models = {} if model=='Monodepth' else ''
        self.dataloader = dataloader
        self.dataset= dataset
        self.device = device
        self.pacing_function = pacing_function
        self.sample_scores = []
        self.opt= opt
        self.batch_size=1
        self.num_scales = len(self.opt.scales) # default=[0], we only perform single scale training
        self.num_input_frames = len(self.opt.frame_ids) # default=[0, -1, 1]
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames # default=2
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        
        
        if model=='Monodepth' and not os.path.exists("/home/jturriatellallire/scores_kitti_self.npy"):

            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, False)
                                    
            self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)

            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    False,
                    num_input_images=self.num_pose_frames)

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn": #USED BY SPIDEPTH
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)


            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Checkpoint not found at {model_path}")
            
            encoder_path = os.path.join(model_path, "encoder.pth")
            decoder_path = os.path.join(model_path, "depth.pth")
            pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
            pose_path = os.path.join(model_path, "pose.pth")


            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
            loaded_dict_enc = self.remove_module_prefix(loaded_dict_enc)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["encoder"].state_dict()}
            self.models["encoder"].load_state_dict(filtered_dict_enc)

            loaded_dict_enc = torch.load(decoder_path, map_location=self.device)
            loaded_dict_enc = self.remove_module_prefix(loaded_dict_enc)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["depth"].state_dict()}
            self.models["depth"].load_state_dict(filtered_dict_enc)

            loaded_dict_enc = torch.load(pose_encoder_path, map_location=self.device)
            loaded_dict_enc = self.remove_module_prefix(loaded_dict_enc)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["pose_encoder"].state_dict()}
            self.models["pose_encoder"].load_state_dict(filtered_dict_enc)

            loaded_dict_enc = torch.load(pose_path, map_location=self.device)
            loaded_dict_enc = self.remove_module_prefix(loaded_dict_enc)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.models["pose"].state_dict()}
            self.models["pose"].load_state_dict(filtered_dict_enc)

            for model in self.models.values():
                model.to(device)

            for m in self.models.values():
                m.eval()


    def pacing(self, step, total_steps, total_samples):
        """
        Define how many samples to use at this stage of training.
        Supports linear, quadratic, exponential, logarithmic, and step pacing functions.
        """
        Nb = int(total_samples * self.opt.b)  # fraction of full training data
        aT = self.opt.a * total_steps  # parameter a times total epochs
        t =step + 1  # current step (1-based index)
        pacing_result=0

        if self.pacing_function == "linear":
            pacing_result= int(Nb + ((1 - self.opt.b) * total_samples / aT) * t)
        elif self.pacing_function == "quadratic":
            pacing_result= int(Nb + (total_samples * (1 - self.opt.b) / aT) * (t ** self.opt.p))
        elif self.pacing_function == "exponential":
            pacing_result= int(Nb + (total_samples * (1 - self.opt.b) / (np.exp(10) - 1)) * (np.exp(10 * t / aT) - 1))
        elif self.pacing_function == "logarithmic":
            pacing_result= int(Nb + total_samples * (1 - self.opt.b) * (1 + (1 / 10) * np.log(t / aT + np.exp(-10))))
        elif self.pacing_function == "step":
            pacing_result= int(Nb + total_samples * (0 if (t / aT)< 1 else 1))
        else:
            raise NotImplementedError(f"Pacing function '{self.pacing_function}' not implemented")
        
        return min(pacing_result, total_samples)

    def get_curriculum_batches(self, step, total_steps, batch_size, score_path="sample_scores.npy"):
        """
        Return a DataLoader for the current step based on the pacing function and stored scores
        """
        if len(self.sample_scores) == 0:
            self.load_scores(score_path)
        
        # Determine the number of samples to use at this stage of training
        selected_size = self.pacing(step, total_steps, len(self.sample_scores))
        selected_indices = self.sorted_indices[:selected_size]

        # Create a subset of the dataset based on selected indices
        selected_subset = torch.utils.data.Subset(self.dataloader.dataset, selected_indices)
        # Create a DataLoader for the selected subset
        selected_loader = torch.utils.data.DataLoader(
            selected_subset, batch_size=batch_size, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True
        )

        return selected_loader
    
    def score_and_save_losses(self, score_path="sample_scores.npy"):
        """
        Computes and stores difficulty scores (losses) of the dataset.
        Only needs to be run once before curriculum training begins.
        """
        sample_losses = []
        new_dataloader= torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True, drop_last=False
        )

        with torch.no_grad():
            for inputs in new_dataloader:
                for key in inputs:
                    inputs[key] = inputs[key].to(self.device)
                outputs, losses = self.process_batch(inputs)
                total_loss = losses["loss"].item()
                sample_losses.append(total_loss)

        sample_losses = np.array(sample_losses)
        np.save(score_path, sample_losses)
        print(f"Saved difficulty scores to: {score_path}")

        for model in self.models.values():
            del model
        torch.cuda.empty_cache()


    def load_scores(self, score_path="sample_scores.npy"):
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Score file {score_path} not found. Run score_and_save_losses() first.")
        self.sample_scores = np.load(score_path)
        self.sorted_indices = np.argsort(self.sample_scores)  # easiest samples first


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared": # default no
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

        if self.opt.predictive_mask: # default no
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
        
        outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
    
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    # print(axisangle.shape)
                    # axisangle:[12, 1, 1, 3]  translation:[12, 1, 1, 3]
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
                    # outputs[("cam_T_cam", 0, f_i)]: [12, 4, 4]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

                #depth = disp
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn" and not self.opt.use_stereo:

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                # pix_coords: [bs, h, w, 2]

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True)

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
                    

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                #weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).to(self.device))
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            #Save raw reprojection loss (mean over batch)
            losses["reproj_loss"] = reprojection_loss.mean()

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001 #.cuda() was replaced with to(self.device)

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            if color.shape[-2:] != disp.shape[-2:]:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            # if GPU memory is not enough, you can downsample color instead
            # color = F.interpolate(color, [self.opt.height // 2, self.opt.width // 2], mode="bilinear", align_corners=False)
            smooth_loss = 0
            smooth_loss = get_smooth_loss(norm_disp, color)
            # smooth_loss

             # Save smoothness loss per scale
            losses["smooth_loss"] = self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        _, _, h_gt, w_gt = inputs["depth_gt"].shape #Added
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [h_gt, w_gt], mode="bilinear", align_corners=False), 1e-3, 80)
        #depth_pred = torch.clamp(F.interpolate(
        #    depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        if self.opt.dataset=="kitti":
            crop_mask = torch.zeros_like(mask)
            crop_mask[:, :, 153:371, 44:1197] = 1
            mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def remove_module_prefix(self, state_dict):
        return {k.replace("module.", ""): v for k, v in state_dict.items()}