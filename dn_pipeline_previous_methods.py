    ### previous methods
    ############################################################################################################
    # start: 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss + lseg loss
    ############################################################################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view every secret_edit_rate steps
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1, lpips, regularization

    #         # compute masked lpips value
    #         mask_np = self.ip2p_ptd.mask
    #         # Convert mask to tensor and ensure it's the right shape/device
    #         mask_tensor = torch.from_numpy(mask_np).float()
    #         if len(mask_tensor.shape) == 2:
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #         if mask_tensor.shape[0] == 1:
    #             mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #         mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #         # Prepare model output
    #         model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #         # Apply mask to both images
    #         masked_model_rgb = model_rgb_secret * mask_tensor
    #         masked_ref_image = self.ref_image_tensor * mask_tensor

    #         # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #         ref_loss = self.lpips_loss_fn(
    #             masked_model_rgb,
    #             masked_ref_image
    #         ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #         ref_l1_loss = torch.nn.functional.l1_loss(
    #             masked_model_rgb,
    #             masked_ref_image
    #         )
    #         # loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss
    #         metrics_dict["ref_loss"] = self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss
    #         loss_dict["ref_loss"] = self.config_secret.ref_loss_weight * ref_loss + ref_l1_loss

    #         # edge loss
    #         # rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #         # edge_loss = self.edge_loss_fn(
    #         #     rendered_image_secret.to(self.config_secret.device), 
    #         #     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #         #     self.original_secret_edges.to(self.config_secret.device),
    #         #     image_dir,
    #         #     step
    #         # )
    #         # loss_dict_secret["main_loss"] += edge_loss

    #         # lseg loss
    #         rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #         # [1, 512, 256, 256]
    #         rendered_image_sem_feature = self.lseg_model.get_image_features(rendered_image_secret.to(self.config_secret.device))
    #         rendered_image_logits = self.lseg_model.decode_feature(rendered_image_sem_feature)
    #         rendered_semantic = self.lseg_model.visualize_sem(rendered_image_logits) # (c, h, w)
    #         if step % 50 == 0:
    #             save_image(rendered_semantic, image_dir / f'{step}_rendered_semantic.png')

    #         # l1 loss
    #         # lseg_loss = torch.nn.functional.l1_loss(rendered_image_sem_feature, self.original_image_sem_feature)
    #         # cross loss
    #         lseg_loss = (1 - torch.nn.functional.cosine_similarity(rendered_image_sem_feature, self.original_image_sem_feature, dim=1)).mean()
            
    #         # loss_dict_secret["main_loss"] += lseg_loss
    #         metrics_dict["lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
    #         loss_dict["lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight

    #         if step % 100 == 0:
    #             image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         # put the secret metrics and loss into the main dict
    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
    ############################################################################################################
    # end: 2nd stage: secret + non-secret loss + masked fighting (ref, original) loss + lseg loss
    ############################################################################################################


    # 2nd stage: only masked fighting (ref, original) loss
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1 + lpips
    #     loss_dict_secret["main_loss"] = 0.0

    #     # compute masked lpips value
    #     mask_np = self.ip2p_ptd.mask
    #     # Convert mask to tensor and ensure it's the right shape/device
    #     mask_tensor = torch.from_numpy(mask_np).float()
    #     if len(mask_tensor.shape) == 2:
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #     if mask_tensor.shape[0] == 1:
    #         mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #     mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #     mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #     # Prepare model output
    #     model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #     original_image_secret = (self.datamanager.original_cached_train[self.config_secret.secret_view_idx]["image"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1).to(self.ref_image_tensor.device)

    #     # Apply mask to both images
    #     masked_model_rgb = model_rgb_secret * mask_tensor
    #     masked_ref_image = self.ref_image_tensor * mask_tensor
    #     masked_original_image = original_image_secret * mask_tensor

    #     # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #     ref_loss = self.lpips_loss_fn(
    #         masked_model_rgb,
    #         masked_ref_image
    #     ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #     original_loss = self.lpips_loss_fn(
    #         masked_model_rgb,
    #         masked_original_image
    #     ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
        
    #     loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss + original_loss

    #     if step % 100 == 0:
    #         image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #     return model_outputs_secret, loss_dict_secret, metrics_dict_secret
    
    

    # ###################################################################################################
    # # start: 3rd stage: dataset downsampling + only editing with ip2p + masked edge loss + lseg loss
    # ###################################################################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_3rd_stage_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # replace the original dataset with current rendering
    #     if self.first_step:
    #         self.first_step = False
        
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             rendered_image = model_outputs["rgb"].detach()

    #             self.datamanager.original_cached_train[idx]["image"] = rendered_image
    #             self.datamanager.cached_train[idx]["image"] = rendered_image
    #             data["image"] = rendered_image

    #         print("dataset replacement complete!")

    #         # dataset downsampling and return the new secret idx
    #         self.config_secret.secret_view_idx = self.datamanager.downsample_dataset(self.config_secret.downsample_factor, self.config_secret.secret_view_idx)
    #         # save original semantic map
    #         original_image_logits = self.lseg_model.decode_feature(self.original_image_sem_feature)
    #         original_semantic = self.lseg_model.visualize_sem(original_image_logits) # (c, h, w)
    #         save_image(original_semantic, image_dir / f'{step}_original_semantic.png')
    #         save_image(self.original_image_secret, image_dir / f'{step}_original_image.png')

    #     # start editing
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     # ordinary editing
    #     if (not self.makeSequentialEdits):
    #         all_indices = np.arange(len(self.datamanager.cached_train))

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(all_indices)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             if idx == self.config_secret.secret_view_idx:
    #                 image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             ############################ prepare secret rendering ######################## 
    #             model_outputs_secret = self.model(self.camera_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             ############################ prepare secret rendering ######################## 

    #             ############################ for edge loss ########################
    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Apply mask to both images
    #             masked_model_rgb = rendered_image_secret * mask_tensor

    #             edge_loss = self.edge_loss_fn(
    #                 masked_model_rgb.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step,
    #                 mask_tensor
    #             )

    #             metrics_dict["secret_edge_loss"] = edge_loss * self.config_secret.edge_loss_weight
    #             loss_dict["secret_edge_loss"] = edge_loss * self.config_secret.edge_loss_weight
    #             ############################ for edge loss ########################

    #             ############################ for lseg loss ########################
    #             [1, 512, 256, 256]
    #             rendered_image_sem_feature = self.lseg_model.get_image_features(rendered_image_secret.to(self.config_secret.device))
    #             rendered_image_logits = self.lseg_model.decode_feature(rendered_image_sem_feature)
    #             rendered_semantic = self.lseg_model.visualize_sem(rendered_image_logits) # (c, h, w)
    #             if step % 50 == 0:
    #                 save_image(rendered_semantic, image_dir / f'{step}_rendered_semantic.png')

    #             # l1 loss
    #             # lseg_loss = torch.nn.functional.l1_loss(rendered_image_sem_feature, self.original_image_sem_feature)
    #             # cross loss
    #             lseg_loss = (1 - torch.nn.functional.cosine_similarity(rendered_image_sem_feature, self.original_image_sem_feature, dim=1)).mean()

    #             metrics_dict["secret_lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
    #             loss_dict["secret_lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
    #             ############################ for lseg loss ########################

    #         # sequential editing
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 25 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
    #         if idx == self.config_secret.secret_view_idx:
    #             image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    # ###################################################################################################
    # # end: 3rd stage: dataset downsampling + only editing with ip2p + masked edge loss + lseg loss
    # ###################################################################################################

    ######################################################
    # start: 3rd stage: only editing with ip2p + lseg loss
    ######################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_3rd_stage_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # replace the original dataset with current rendering
    #     if self.first_step:
    #         self.first_step = False
        
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             rendered_image = model_outputs["rgb"].detach()

    #             self.datamanager.original_cached_train[idx]["image"] = rendered_image
    #             self.datamanager.cached_train[idx]["image"] = rendered_image
    #             data["image"] = rendered_image

    #         print("dataset replacement complete!")

    #     # start editing
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     # ordinary editing
    #     if (not self.makeSequentialEdits):
    #         all_indices = np.arange(len(self.datamanager.cached_train))

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(all_indices)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             if idx == self.config_secret.secret_view_idx:
    #                 image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
                
    #             ############################ for edge loss ########################
    #             # model_outputs_secret = self.model(self.camera_secret)
    #             # metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             # loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1 + lpips

    #             # # edge loss
    #             # rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             # edge_loss = self.edge_loss_fn(
    #             #     rendered_image_secret.to(self.config_secret.device), 
    #             #     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #             #     self.original_secret_edges.to(self.config_secret.device),
    #             #     image_dir,
    #             #     step + 1
    #             # )
    #             # loss_dict_secret["main_loss"] += edge_loss * 0.1

    #             # # put the secret metrics and loss into the main dict
    #             # for k, v in metrics_dict_secret.items():
    #             #     metrics_dict[f"secret_{k}"] = v
    #             # for k, v in loss_dict_secret.items():
    #             #     loss_dict[f"secret_{k}"] = v
    #             ############################ for edge loss ########################

    #             ############################ for lseg loss ########################
    #             model_outputs_secret = self.model(self.camera_secret)

    #             # lseg loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             # [1, 512, 256, 256]
    #             rendered_image_sem_feature = self.lseg_model.get_image_features(rendered_image_secret.to(self.config_secret.device))
                
    #             # l1 loss
    #             # lseg_loss = torch.nn.functional.l1_loss(rendered_image_sem_feature, self.original_image_sem_feature)
    #             # cross loss
    #             lseg_loss = (1 - torch.nn.functional.cosine_similarity(rendered_image_sem_feature, self.original_image_sem_feature, dim=1)).mean()

    #             metrics_dict["secret_lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
    #             loss_dict["secret_lseg_loss"] = lseg_loss * self.config_secret.lseg_loss_weight
    #             ############################ for lseg loss ########################

    #         # sequential editing
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 25 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
    #         if idx == self.config_secret.secret_view_idx:
    #             image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################
    # end: 3rd stage: only editing with ip2p + lseg loss
    ######################################################

    ######################################################
    # start: 3rd stage: only editing with ip2p (+ edge loss)
    ######################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_3rd_stage_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # replace the original dataset with current rendering
    #     if self.first_step:
    #         self.first_step = False
        
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             rendered_image = model_outputs["rgb"].detach()

    #             self.datamanager.original_cached_train[idx]["image"] = rendered_image
    #             self.datamanager.cached_train[idx]["image"] = rendered_image
    #             data["image"] = rendered_image

    #         print("dataset replacement complete!")

    #     # start editing
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         all_indices = np.arange(len(self.datamanager.cached_train))

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(all_indices)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             if idx == self.config_secret.secret_view_idx:
    #                 image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
                
    #             ############################ for edge loss ########################
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret) # l1 + lpips

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step + 1
    #             )
    #             loss_dict_secret["main_loss"] += edge_loss * 0.1

    #             # put the secret metrics and loss into the main dict
    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
    #             ############################ for edge loss ########################

    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 25 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
    #         if idx == self.config_secret.secret_view_idx:
    #             image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################
    # end: 3rd stage: only editing with ip2p (+ edge loss)
    ######################################################

    ######################################################
    # start: 3rd stage: original IGS2GS
    ######################################################
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_3rd_stage_only_editing_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # replace the original dataset with current rendering
    #     if self.first_step:
    #         self.first_step = False
        
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             rendered_image = model_outputs["rgb"].detach()

    #             self.datamanager.original_cached_train[idx]["image"] = rendered_image
    #             self.datamanager.cached_train[idx]["image"] = rendered_image
    #             data["image"] = rendered_image

    #         print("dataset replacement complete!")

    #     # start editing
    #     if (step % self.config.gs_steps) == 0:
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 25 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')
            
    #         if idx == self.config_secret.secret_view_idx:
    #             image_save_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False

    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################
    # end: 3rd stage: original IGS2GS
    ######################################################
    
    # __camera_pose_offset_updating__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_camera_pose_offset_updating"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     if self.first_step:
    #         self.first_step = False
    #         # find the best secret view that align with the reference image
    #         current_secret_idx = 0
    #         current_score = float("inf")
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Prepare model output
    #             model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #             # Apply mask to both images
    #             masked_model_rgb = model_rgb * mask_tensor
    #             masked_ref_image = self.ref_image_tensor * mask_tensor

    #             # Compute masked LPIPS score
    #             lpips_score = self.lpips_loss_fn(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             l1_score = torch.nn.functional.l1_loss(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             # score = l1_score
    #             score = lpips_score

    #             # # unmasked lpips score
    #             # lpips_score = self.lpips_loss_fn(
    #             #     (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #             #     self.ref_image_tensor
    #             # ).item()

    #             if score < current_score:
    #                 current_score = score
    #                 current_secret_idx = idx

    #         self.secret_view_idx = current_secret_idx
    #         camera, data = self.datamanager.next_train_idx(self.secret_view_idx)
    #         model_outputs = self.model(camera)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image(rendered_image, image_dir / f"best_secret_view_{self.secret_view_idx}_score_{current_score}.png")

    #         CONSOLE.print(f"Best secret view index: {self.secret_view_idx} with score: {current_score}")

    #         # start pose offset updating
    #         # Get the secret view camera and initialize pose offset parameters
    #         camera_secret, data_secret = self.datamanager.next_train_idx(self.secret_view_idx)

    #         original_pose_backup = camera_secret.camera_to_worlds.clone()
            
    #         # Initialize camera pose offset parameters (6DOF: translation + rotation)
    #         if not hasattr(self, 'camera_pose_offset'):
    #             # Translation offset (x, y, z)
    #             self.translation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
    #             # Rotation offset (axis-angle representation)
    #             self.rotation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
                
    #             # Optimizer for camera pose offset
    #             self.pose_optimizer = torch.optim.Adam([self.translation_offset, self.rotation_offset], lr=float(self.config_secret.pose_learning_rate))

    #         # Before the pose optimization loop, store the gradient state and disable model gradients
    #         model_param_grad_states = {}
    #         for name, param in self.model.named_parameters():
    #             model_param_grad_states[name] = param.requires_grad
    #             param.requires_grad = False

    #         # Camera pose optimization loop
    #         num_pose_iterations = self.config_secret.num_pose_iterations
            
    #         for pose_iter in range(num_pose_iterations):
    #             self.pose_optimizer.zero_grad()
                
    #             # Create rotation matrix from axis-angle representation
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             # Apply rotation offset: R_new = R_offset @ R_original
    #             original_rotation = original_pose_backup[0, :3, :3]
    #             new_rotation = rotation_matrix @ original_rotation
                
    #             # Apply translation offset: t_new = t_original + t_offset
    #             original_translation = original_pose_backup[0, :3, 3]
    #             new_translation = original_translation + self.translation_offset
                
    #             # Construct new camera-to-world matrix
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
                
    #             # Render with updated camera pose
    #             with torch.enable_grad():
    #                 model_outputs = self.model(camera_secret)
                    
    #                 # Compute LPIPS loss with mask
    #                 mask_np = self.ip2p_ptd.mask
    #                 mask_tensor = torch.from_numpy(mask_np).float()
    #                 if len(mask_tensor.shape) == 2:
    #                     mask_tensor = mask_tensor.unsqueeze(0)
    #                 if mask_tensor.shape[0] == 1:
    #                     mask_tensor = mask_tensor.repeat(3, 1, 1)
    #                 mask_tensor = mask_tensor.unsqueeze(0)
    #                 mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #                 model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1) # don't detach here, or the gradients won't exist
    #                 masked_model_rgb = model_rgb * mask_tensor
    #                 masked_ref_image = self.ref_image_tensor * mask_tensor
                    
    #                 lpips_loss = self.lpips_loss_fn(masked_model_rgb, masked_ref_image)
    #                 l1_loss = torch.nn.functional.l1_loss(masked_model_rgb, masked_ref_image)
                    
    #                 # Add regularization to prevent large offsets
    #                 translation_reg = torch.norm(self.translation_offset) * float(self.config_secret.translation_reg_weight)
    #                 rotation_reg = torch.norm(self.rotation_offset) * float(self.config_secret.rotation_reg_weight)
    #                 # total_loss = lpips_loss + translation_reg + rotation_reg
    #                 total_loss = lpips_loss
    #                 # total_loss = l1_loss
                    
    #                 # Backward pass and optimization step
    #                 total_loss.backward()
    #                 self.pose_optimizer.step()
                
    #             # Optional: clamp offsets to reasonable ranges
    #             with torch.no_grad():
    #                 self.translation_offset.clamp_(-self.config_secret.max_translation_offset, 
    #                                             self.config_secret.max_translation_offset)
    #                 self.rotation_offset.clamp_(-self.config_secret.max_rotation_offset, 
    #                                         self.config_secret.max_rotation_offset)
                
    #             if pose_iter % 50 == 0:
    #                 with torch.no_grad():
    #                     CONSOLE.print(
    #                         # f"Translation gradient norm: {self.translation_offset.grad.norm().item()}",
    #                         # f"Rotation gradient norm: {self.rotation_offset.grad.norm().item()}",
    #                         f"Pose iter {pose_iter}: total loss = {total_loss.item():.6f}, "
    #                         f"Trans offset norm = {torch.norm(self.translation_offset).item():.6f}, "
    #                         f"Rot offset norm = {torch.norm(self.rotation_offset).item():.6f}"
    #                     )

    #                     rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                    
    #                     new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #                     new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                        
    #                     new_c2w = original_pose_backup.clone()
    #                     new_c2w[0, :3, :3] = new_rotation
    #                     new_c2w[0, :3, 3] = new_translation
                        
    #                     camera_secret.camera_to_worlds = new_c2w
    #                     optimized_camera = camera_secret
                        
    #                     final_outputs = self.model(optimized_camera)
    #                     rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                        
    #                     # Compute final LPIPS score
    #                     model_rgb = (final_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
    #                     masked_model_rgb = model_rgb * mask_tensor
                        
    #                     save_image(rendered_image, 
    #                             image_dir / f"optimized_secret_view_{self.secret_view_idx}_step_{pose_iter}_loss_{total_loss.item():.6f}.png")
            
    #         # After optimization, restore model parameter gradient states
    #         for name, param in self.model.named_parameters():
    #             param.requires_grad = model_param_grad_states[name]

    #         # # Final rendering with optimized pose
    #         # if step % 10 == 0:
    #         #     with torch.no_grad():
    #         #         rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                    
    #         #         new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #         #         new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                    
    #         #         new_c2w = original_pose_backup.clone()
    #         #         new_c2w[0, :3, :3] = new_rotation
    #         #         new_c2w[0, :3, 3] = new_translation
                    
    #         #         camera_secret.camera_to_worlds = new_c2w
    #         #         optimized_camera = camera_secret
                    
    #         #         final_outputs = self.model(optimized_camera)
    #         #         rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                    
    #         #         # Compute final LPIPS score
    #         #         model_rgb = (final_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
    #         #         masked_model_rgb = model_rgb * mask_tensor
    #         #         final_lpips = self.lpips_loss_fn(masked_model_rgb, masked_ref_image).item()
                    
    #         #         save_image(rendered_image, 
    #         #                 image_dir / f"optimized_secret_view_{self.secret_view_idx}_step_{step}_lpips_{final_lpips:.6f}.png")
                    
    #         #         CONSOLE.print(f"Final optimized LPIPS score: {final_lpips:.6f}")
                    
    #         #         # Store optimized camera for use in training
    #         #         self.camera_secret = optimized_camera
    #         #         self.data_secret = data_secret       

    #         # secret data preparation
    #         with torch.no_grad():
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #             new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
    #             optimized_camera = camera_secret
                
    #             final_outputs = self.model(optimized_camera)
    #             rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)  
                
    #             # secret data preparation
    #             self.camera_secret, self.data_secret = optimized_camera, data_secret
    #             self.original_image_secret = rendered_image
    #             self.depth_image_secret = self.datamanager.original_cached_train[self.secret_view_idx]["depth"] # [bs, h, w]
    #             # original secret edges
    #             self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)
            
    #     return model_outputs, loss_dict, metrics_dict
    ######################################################

    # only secret
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # # only secret loss
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     torch.cuda.empty_cache()

    #     return model_outputs_secret, loss_dict_secret, metrics_dict_secret
        ######################################################


    # secret + non-secret loss + masked fighting (ref, original) loss
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_loss_fighting_ref_original_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view every secret_edit_rate steps
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #         # compute masked lpips value
    #         mask_np = self.ip2p_ptd.mask
    #         # Convert mask to tensor and ensure it's the right shape/device
    #         mask_tensor = torch.from_numpy(mask_np).float()
    #         if len(mask_tensor.shape) == 2:
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #         if mask_tensor.shape[0] == 1:
    #             mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #         mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #         mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #         # Prepare model output
    #         model_rgb_secret = (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #         # Apply mask to both images
    #         masked_model_rgb = model_rgb_secret * mask_tensor
    #         masked_ref_image = self.ref_image_tensor * mask_tensor

    #         # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #         ref_loss = self.lpips_loss_fn(
    #             masked_model_rgb,
    #             masked_ref_image
    #         ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #         loss_dict_secret["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

    #         if step % 100 == 0:
    #             image_save_secret = torch.cat([model_outputs_secret["rgb"].detach().permute(2, 0, 1).unsqueeze(0), ((masked_ref_image + 1) / 2).to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')

    #         # put the secret metrics and loss into the main dict
    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # only secret + fighting loss (ref， PTD)
    # def get_train_loss_dict(self, step: int):
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_fighting_loss"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     # a content loss of ref image added to the original rgb (L1 + lpips loss)
    #     ref_loss_weight = 0.2
    #     ref_loss = self.lpips_loss_fn(
    #         (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #         self.ref_image_tensor
    #     ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #     loss_dict["main_loss"] += ref_loss_weight * ref_loss

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # pie + only secret
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_pie_only_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         # ###########################################
    #         # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #         # edited_image_secret is B, C, H, W in [0, 1]
    #         edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #         edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #         # Convert image_original to numpy
    #         if hasattr(self.ip2p_ptd.image_original, 'cpu'):
    #             # It's a PyTorch tensor
    #             image_original_np = self.ip2p_ptd.image_original.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             image_original_np = (image_original_np * 255).astype(np.uint8)
    #         elif hasattr(self.ip2p_ptd.image_original, 'save'):
    #             # It's a PIL Image
    #             image_original_np = np.array(self.ip2p_ptd.image_original)
    #             # PIL images are usually already in [0, 255] uint8 format
    #             if image_original_np.dtype != np.uint8:
    #                 image_original_np = image_original_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             image_original_np = self.ip2p_ptd.image_original

    #         # Convert mask to numpy
    #         if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #             # It's a PyTorch tensor
    #             mask_tensor = self.ip2p_ptd.mask
    #             if mask_tensor.dim() == 4:
    #                 mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #             elif mask_tensor.dim() == 3:
    #                 mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #             else:
    #                 mask_np = mask_tensor.cpu().numpy()
    #             mask_np = (mask_np * 255).astype(np.uint8)
    #         elif hasattr(self.ip2p_ptd.mask, 'save'):
    #             # It's a PIL Image
    #             mask_np = np.array(self.ip2p_ptd.mask)
    #             # Convert to grayscale if needed
    #             if mask_np.ndim == 3:
    #                 mask_np = mask_np[:, :, 0]  # Take first channel
    #             # Ensure it's uint8
    #             if mask_np.dtype != np.uint8:
    #                 if mask_np.max() <= 1.0:
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 else:
    #                     mask_np = mask_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             mask_np = self.ip2p_ptd.mask

    #         # Call the original opencv_seamless_clone function with numpy arrays
    #         result_np = opencv_seamless_clone(edited_image_np, image_original_np, mask_np)

    #         # Convert the result back to PyTorch tensor format
    #         # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #         edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #         edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #         edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #         # ###########################################

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # pie + only_secret + edge loss
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_pie_only_secret_edge_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     # compute edge loss
    #     # rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #     rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #     edge_loss = self.edge_loss_fn(
    #         rendered_image_secret.to(self.config_secret.device), 
    #         self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #         self.original_secret_edges.to(self.config_secret.device),
    #         # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #         image_dir,
    #         step
    #     )
    #     loss_dict["main_loss"] += edge_loss

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')
            
    #         if step % 200 == 0:
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         # ###########################################
    #         # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #         # edited_image_secret is B, C, H, W in [0, 1]
    #         edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #         edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #         # Convert image_original to numpy
    #         if hasattr(self.original_image_secret, 'cpu'):
    #             # It's a PyTorch tensor
    #             image_original_np = self.original_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             image_original_np = (image_original_np * 255).astype(np.uint8)
    #         elif hasattr(self.original_image_secret, 'save'):
    #             # It's a PIL Image
    #             image_original_np = np.array(self.original_image_secret)
    #             # PIL images are usually already in [0, 255] uint8 format
    #             if image_original_np.dtype != np.uint8:
    #                 image_original_np = image_original_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             image_original_np = self.original_image_secret

    #         # Convert mask to numpy
    #         if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #             # It's a PyTorch tensor
    #             mask_tensor = self.ip2p_ptd.mask
    #             if mask_tensor.dim() == 4:
    #                 mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #             elif mask_tensor.dim() == 3:
    #                 mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #             else:
    #                 mask_np = mask_tensor.cpu().numpy()
    #             mask_np = (mask_np * 255).astype(np.uint8)
    #         elif hasattr(self.ip2p_ptd.mask, 'save'):
    #             # It's a PIL Image
    #             mask_np = np.array(self.ip2p_ptd.mask)
    #             # Convert to grayscale if needed
    #             if mask_np.ndim == 3:
    #                 mask_np = mask_np[:, :, 0]  # Take first channel
    #             # Ensure it's uint8
    #             if mask_np.dtype != np.uint8:
    #                 if mask_np.max() <= 1.0:
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 else:
    #                     mask_np = mask_np.astype(np.uint8)
    #         else:
    #             # If it's already numpy, just use it
    #             mask_np = self.ip2p_ptd.mask

    #         # Call the original opencv_seamless_clone function with numpy arrays
    #         result_np = opencv_seamless_clone(edited_image_np, image_original_np, mask_np)

    #         # Convert the result back to PyTorch tensor format
    #         # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #         edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #         edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #         edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #         # ###########################################

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         if step % 200 == 0:
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # remove close views + only secret
    # def _is_view_close_to_secret(self, camera):
    #     """
    #     Check if the current camera view is close to the secret view.
    #     You can customize this logic based on your specific criteria.
    #     """
    #     # Get camera poses
    #     current_pose = camera.camera_to_worlds[0]  # Assuming batch size 1
    #     secret_pose = self.camera_secret.camera_to_worlds[0]
        
    #     # Calculate distance between camera positions
    #     position_distance = torch.norm(current_pose[:3, 3] - secret_pose[:3, 3])
        
    #     # Calculate angular difference between camera orientations
    #     # Using the rotation matrices (first 3x3 of the poses)
    #     current_rotation = current_pose[:3, :3]
    #     secret_rotation = secret_pose[:3, :3]
        
    #     # Calculate rotation difference using trace of R1^T * R2
    #     rotation_diff = torch.trace(torch.matmul(current_rotation.T, secret_rotation))
    #     # Convert to angle: cos(angle) = (trace(R) - 1) / 2
    #     angle_diff = torch.acos(torch.clamp((rotation_diff - 1) / 2, -1, 1))

    #     # print("position_distance: ", position_distance, "angle_diff: ", angle_diff)
        
    #     # Define thresholds (you may need to adjust these based on your scene scale)
    #     position_threshold = 1.0  # Adjust based on your scene scale
    #     angle_threshold = 0.5  # Radians (0.2 about 11.5 degrees, 0.5, 60 degrees)
        
    #     # Check if view is close based on both position and orientation
    #     is_close = (position_distance < position_threshold) and (angle_diff < angle_threshold)
        
    #     return is_close

    # def _is_rgb_loss(self, loss_key):
    #     """
    #     Determine if a loss key corresponds to an RGB-related loss.
    #     Based on your loss_dict structure: {'main_loss': ..., 'scale_reg': ...}
        
    #     Simple approach: Only filter main_loss for close views
    #     """
    #     # Only filter out main_loss for close views, keep everything else
    #     return loss_key == 'main_loss'
    
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_only_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
        
    #     # Check if current view is close to secret view
    #     is_close_to_secret = self._is_view_close_to_secret(camera)
        
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
        
    #     # Filter out RGB loss for views close to secret view
    #     if is_close_to_secret:
    #         # Remove RGB-related losses while keeping depth and normal losses
    #         filtered_loss_dict = {}
    #         for key, value in loss_dict.items():
    #             # Keep all losses except RGB-related ones
    #             if not self._is_rgb_loss(key):
    #                 filtered_loss_dict[key] = value
    #             else:
    #                 # Optionally log that we're skipping this loss
    #                 print(f"Skipping RGB loss '{key}' for view close to secret view")
    #         loss_dict = filtered_loss_dict

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0:
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     torch.cuda.empty_cache()

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # seva + only secret
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_seva_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # non-editing steps loss computing
    #     camera, data = self.datamanager.next_train(step)
    #     model_outputs = self.model(camera)
    #     metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #     loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #     # update the secret view w/o editing (important)
    #     model_outputs_secret = self.model(self.camera_secret)
    #     metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #     loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #     #----------------secret view editing----------------
    #     if step % self.config_secret.secret_edit_rate == 0 or self.first_iter:
    #         self.first_iter = False

    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #             image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #             image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #             secret_idx=0,
    #             depth=self.depth_image_secret,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image_secret.size() != rendered_image_secret.size()):
    #             edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #         self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #         self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)
            
    #         # save edited secret image
    #         # generate ves views
    #         rgb_list = []
    #         for ves_camera in self.ves_cameras:
    #             model_outputs_ves = self.model(ves_camera)
    #             rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2) # [1, 3, H, W]
    #             rgb_list.append(rendered_image_ves.cpu().squeeze())

    #         row1 = torch.cat([rgb_list[8], rgb_list[7], rgb_list[6]], dim=2)
    #         row2 = torch.cat([rgb_list[5], rgb_list[0], rgb_list[4]], dim=2)
    #         row3 = torch.cat([rgb_list[3], rgb_list[2], rgb_list[1]], dim=2)  # concat along W

    #         # Now stack the three rows along H to get a single [3, 3H, 3W] image
    #         img = torch.cat([row1, row2, row3], dim=1)  # concat along H

    #         save_image(img.clamp(0, 1), image_dir / f'{step}_ves_image.png')

    #         # save secret images
    #         image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #         save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #         save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         # seva results
    #         all_imgs_path = [str(image_dir / f'{step}_secret_image.png')] + [None] * self.num_targets

    #         print(all_imgs_path)

    #         # Create image conditioning.
    #         image_cond = {
    #             "img": all_imgs_path,
    #             "input_indices": self.input_indices,
    #             "prior_indices": self.anchor_indices,
    #         }
    #         # Create camera conditioning.
    #         camera_cond = {
    #             "c2w": self.seva_c2ws.clone(),
    #             "K": self.Ks.clone(),
    #             "input_indices": list(range(self.num_inputs + self.num_targets)),
    #         }

    #         # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
    #         video_path_generator = run_one_scene(
    #             self.task,
    #             self.VERSION_DICT,  # H, W maybe updated in run_one_scene
    #             model=self.MODEL,
    #             ae=self.AE,
    #             conditioner=self.CONDITIONER,
    #             denoiser=self.DENOISER,
    #             image_cond=image_cond,
    #             camera_cond=camera_cond,
    #             save_path=image_dir / f'{step}_seva',
    #             use_traj_prior=True,
    #             traj_prior_Ks=self.anchor_Ks,
    #             traj_prior_c2ws=self.anchor_c2ws,
    #             seed=self.seed,
    #         )
    #         for _ in video_path_generator:
    #             pass

    #         # load seva images
    #         # images in 00x.png 's format under image_dir / samples-rgb / f'{step}_seva' folder
    #         self.rgb_list_seva = []
    #         for i in range(self.num_targets + 1):
    #             image_path = image_dir / f'{step}_seva/samples-rgb/00{i}.png'
    #             image = Image.open(image_path).convert('RGB')
    #             transform = transforms.ToTensor()  # Converts PIL to [C, H, W] and [0, 1]
    #             rgb_tensor = transform(image)
    #             self.rgb_list_seva.append(rgb_tensor)

    #         row1 = torch.cat([self.rgb_list_seva[8], self.rgb_list_seva[7], self.rgb_list_seva[6]], dim=2)
    #         row2 = torch.cat([self.rgb_list_seva[5], self.rgb_list_seva[0], self.rgb_list_seva[4]], dim=2)
    #         row3 = torch.cat([self.rgb_list_seva[3], self.rgb_list_seva[2], self.rgb_list_seva[1]], dim=2)  # concat along W

    #         # Now stack the three rows along H to get a single [3, 3H, 3W] image
    #         img = torch.cat([row1, row2, row3], dim=1)  # concat along H

    #         save_image(img.clamp(0, 1), image_dir / f'{step}_ves_seva_image.png')

    #         # Add SEVA images to dataloader
    #         # Update cached_train and create data entries for each SEVA view
    #         for i, (seva_image, ves_camera) in enumerate(zip(self.rgb_list_seva, self.ves_cameras)):
    #             # Convert from [C, H, W] to [H, W, C] for dataloader format
    #             seva_image_hwc = seva_image.permute(1, 2, 0).to(self.config_secret.device).to(self.original_image_secret.dtype)
                
    #             # Get the corresponding VES view index from predefined indices
    #             view_idx = self.ves_view_indices[i]
                
    #             # Update cached_train with SEVA image
    #             self.datamanager.cached_train[view_idx]["image"] = seva_image_hwc
                    
    #     # also update seva views in normal steps
    #     for i, (seva_image, ves_camera) in enumerate(zip(self.rgb_list_seva, self.ves_cameras)):
    #         # Convert from [C, H, W] to [H, W, C] for dataloader format
    #         seva_image_hwc = seva_image.permute(1, 2, 0).to(self.config_secret.device).to(self.original_image_secret.dtype)
            
    #         # Get the corresponding VES view index from predefined indices
    #         view_idx = self.ves_view_indices[i]

    #         # Create data dict for this SEVA view for loss computation
    #         data_seva = {
    #             "image": seva_image_hwc,
    #             "idx": view_idx,
    #             "is_ves_view": True,  # Flag to identify VES views
    #         }

    #         # Get model outputs for this SEVA view
    #         model_outputs_seva = self.model(ves_camera)
            
    #         # Compute metrics and loss for this SEVA view
    #         # Note: We're only computing image-based losses, not depth/normal losses
    #         metrics_dict_seva = self.model.get_metrics_dict(model_outputs_seva, data_seva)
            
    #         # Create a custom loss dict that only includes image-based losses
    #         loss_dict_seva = {}
            
    #         # loss for seva views
    #         # Only compute image-based losses for VES views
    #         if "rgb_loss" in self.model.get_loss_dict(model_outputs_seva, data_seva, metrics_dict_seva):
    #             # Get the full loss dict first
    #             full_loss_dict = self.model.get_loss_dict(model_outputs_seva, data_seva, metrics_dict_seva)
                
    #             # Filter to only include image-based losses (skip depth/normal losses)
    #             for k, v in full_loss_dict.items():
    #                 if any(term in k.lower() for term in ["rgb", "image", "psnr", "ssim", "lpips"]):
    #                     loss_dict_seva[k] = v
    #         else:
    #             # If the model doesn't separate losses, compute a L1 + lpips loss
    #             rgb_pred = model_outputs_seva["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2) # [1, 3, H, W], [0 ,1]
    #             rgb_gt = data_seva["image"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             # loss_dict_seva["rgb_loss"] = torch.nn.functional.mse_loss(rgb_pred, rgb_gt)
    #             loss_dict_seva["rgb_loss"] = torch.nn.functional.l1_loss(rgb_pred, rgb_gt) + 0.1 * self.lpips_loss_fn(2 * rgb_pred - 1, 2 *  rgb_gt - 1) # make them normalized to [-1, 1]
            
    #         # Add to main dicts with unique keys
    #         for k, v in metrics_dict_seva.items():
    #             metrics_dict[f"seva_view_{i}_{k}"] = v
    #         for k, v in loss_dict_seva.items():
    #             loss_dict[f"seva_view_{i}_{k}"] = v

    #     # put the secret metrics and loss into the main dict
    #     for k, v in metrics_dict_secret.items():
    #         metrics_dict[f"secret_{k}"] = v
    #     for k, v in loss_dict_secret.items():
    #         loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # __IGS2GS_IN2N_pie_edge__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 # Convert mask to numpy
    #                 if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                     # It's a PyTorch tensor
    #                     mask_tensor = self.ip2p_ptd.mask
    #                     if mask_tensor.dim() == 4:
    #                         mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                     elif mask_tensor.dim() == 3:
    #                         mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                     else:
    #                         mask_np = mask_tensor.cpu().numpy()
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                     # It's a PIL Image
    #                     mask_np = np.array(self.ip2p_ptd.mask)
    #                     # Convert to grayscale if needed
    #                     if mask_np.ndim == 3:
    #                         mask_np = mask_np[:, :, 0]  # Take first channel
    #                     # Ensure it's uint8
    #                     if mask_np.dtype != np.uint8:
    #                         if mask_np.max() <= 1.0:
    #                             mask_np = (mask_np * 255).astype(np.uint8)
    #                         else:
    #                             mask_np = mask_np.astype(np.uint8)
    #                 else:
    #                     # If it's already numpy, just use it
    #                     mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        #####################################################

    # __IGS2GS_IN2N_pie_edge_camera_pose_offset_updating__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_camera_pose_offset_updating"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     if self.first_step:
    #         self.first_step = False
    #         # find the best secret view that align with the reference image
    #         current_secret_idx = 0
    #         current_score = float("inf")

    #         # Lists to store scores for all views
    #         all_lpips_scores = []
    #         all_l1_scores = []
    #         view_indices = []
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Prepare model output
    #             model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #             # Apply mask to both images
    #             masked_model_rgb = model_rgb * mask_tensor
    #             masked_ref_image = self.ref_image_tensor * mask_tensor

    #             # Compute masked LPIPS score
    #             lpips_score = self.lpips_loss_fn(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             l1_score = torch.nn.functional.l1_loss(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             # Store scores for plotting
    #             all_lpips_scores.append(lpips_score)
    #             all_l1_scores.append(l1_score)
    #             view_indices.append(idx)

    #             score = l1_score
    #             # score = lpips_score

    #             if score < current_score:
    #                 current_score = score
    #                 current_secret_idx = idx

    #         # Create and save score curves
    #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    #         # Plot LPIPS scores
    #         ax1.plot(view_indices, all_lpips_scores, 'b-', linewidth=2, label='LPIPS Score')
    #         ax1.axvline(x=current_secret_idx, color='r', linestyle='--', linewidth=2, 
    #                 label=f'Best View (idx={current_secret_idx})')
    #         ax1.scatter([current_secret_idx], [current_score], color='red', s=100, zorder=5)
    #         ax1.set_xlabel('View Index')
    #         ax1.set_ylabel('LPIPS Score')
    #         ax1.set_title('LPIPS Scores Across All Views')
    #         ax1.grid(True, alpha=0.3)
    #         ax1.legend()

    #         # Plot L1 scores
    #         ax2.plot(view_indices, all_l1_scores, 'g-', linewidth=2, label='L1 Score')
    #         best_l1_idx = view_indices[np.argmin(all_l1_scores)]
    #         best_l1_score = min(all_l1_scores)
    #         ax2.axvline(x=best_l1_idx, color='orange', linestyle='--', linewidth=2, 
    #                 label=f'Best L1 View (idx={best_l1_idx})')
    #         ax2.scatter([best_l1_idx], [best_l1_score], color='orange', s=100, zorder=5)
    #         ax2.set_xlabel('View Index')
    #         ax2.set_ylabel('L1 Score')
    #         ax2.set_title('L1 Scores Across All Views')
    #         ax2.grid(True, alpha=0.3)
    #         ax2.legend()

    #         plt.tight_layout()
    #         plt.savefig(image_dir / 'score_curves_comparison.png', dpi=300, bbox_inches='tight')
    #         plt.close()

    #         self.secret_view_idx = current_secret_idx
    #         # self.secret_view_idx = self.config_secret.secret_view_idx
    #         camera, data = self.datamanager.next_train_idx(self.secret_view_idx)
    #         model_outputs = self.model(camera)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image(rendered_image, image_dir / f"best_secret_view_{self.secret_view_idx}_score_{current_score}.png")

    #         CONSOLE.print(f"Best secret view index: {self.secret_view_idx} with score: {current_score}")

    #         # start pose offset updating
    #         # Get the secret view camera and initialize pose offset parameters
    #         camera_secret, data_secret = self.datamanager.next_train_idx(self.secret_view_idx)

    #         original_pose_backup = camera_secret.camera_to_worlds.clone()
            
    #         # Initialize camera pose offset parameters (6DOF: translation + rotation)
    #         if not hasattr(self, 'camera_pose_offset'):
    #             # Translation offset (x, y, z)
    #             self.translation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
    #             # Rotation offset (axis-angle representation)
    #             self.rotation_offset = torch.zeros(3, device=camera_secret.camera_to_worlds.device, requires_grad=True)
                
    #             # Optimizer for camera pose offset
    #             self.pose_optimizer = torch.optim.Adam([self.translation_offset, self.rotation_offset], lr=float(self.config_secret.pose_learning_rate))

    #         # # Before the pose optimization loop, store the gradient state and disable model gradients
    #         # model_param_grad_states = {}
    #         # for name, param in self.model.named_parameters():
    #         #     model_param_grad_states[name] = param.requires_grad
    #         #     param.requires_grad = False

    #         # Camera pose optimization loop
    #         num_pose_iterations = self.config_secret.num_pose_iterations
            
    #         for pose_iter in range(num_pose_iterations):
    #             self.pose_optimizer.zero_grad()
                
    #             # Create rotation matrix from axis-angle representation
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             # Apply rotation offset: R_new = R_offset @ R_original
    #             original_rotation = original_pose_backup[0, :3, :3]
    #             new_rotation = rotation_matrix @ original_rotation
                
    #             # Apply translation offset: t_new = t_original + t_offset
    #             original_translation = original_pose_backup[0, :3, 3]
    #             new_translation = original_translation + self.translation_offset
                
    #             # Construct new camera-to-world matrix
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
                
    #             # Render with updated camera pose
    #             with torch.enable_grad():
    #                 model_outputs = self.model(camera_secret)
                    
    #                 # Compute LPIPS loss with mask
    #                 mask_np = self.ip2p_ptd.mask
    #                 mask_tensor = torch.from_numpy(mask_np).float()
    #                 if len(mask_tensor.shape) == 2:
    #                     mask_tensor = mask_tensor.unsqueeze(0)
    #                 if mask_tensor.shape[0] == 1:
    #                     mask_tensor = mask_tensor.repeat(3, 1, 1)
    #                 mask_tensor = mask_tensor.unsqueeze(0)
    #                 mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #                 model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1) # don't detach here, or the gradients won't exist
    #                 masked_model_rgb = model_rgb * mask_tensor
    #                 masked_ref_image = self.ref_image_tensor * mask_tensor
                    
    #                 lpips_loss = self.lpips_loss_fn(masked_model_rgb, masked_ref_image)
    #                 l1_loss = torch.nn.functional.l1_loss(masked_model_rgb, masked_ref_image)
                    
    #                 # Add regularization to prevent large offsets
    #                 translation_reg = torch.norm(self.translation_offset) * float(self.config_secret.translation_reg_weight)
    #                 rotation_reg = torch.norm(self.rotation_offset) * float(self.config_secret.rotation_reg_weight)
    #                 total_loss = lpips_loss + translation_reg + rotation_reg
    #                 # total_loss = lpips_loss
    #                 # total_loss = l1_loss  + translation_reg + rotation_reg
                    
    #                 # Backward pass and optimization step
    #                 total_loss.backward()
    #                 self.pose_optimizer.step()
                
    #             # Optional: clamp offsets to reasonable ranges
    #             with torch.no_grad():
    #                 self.translation_offset.clamp_(-self.config_secret.max_translation_offset, 
    #                                             self.config_secret.max_translation_offset)
    #                 self.rotation_offset.clamp_(-self.config_secret.max_rotation_offset, 
    #                                         self.config_secret.max_rotation_offset)
                
    #             if pose_iter % 200 == 0 or pose_iter == num_pose_iterations - 1:
    #                 with torch.no_grad():
    #                     CONSOLE.print(
    #                         # f"Translation gradient norm: {self.translation_offset.grad.norm().item()}",
    #                         # f"Rotation gradient norm: {self.rotation_offset.grad.norm().item()}",
    #                         f"Pose iter {pose_iter}: total loss = {total_loss.item():.6f}, "
    #                         f"Trans offset norm = {torch.norm(self.translation_offset).item():.6f}, "
    #                         f"Rot offset norm = {torch.norm(self.rotation_offset).item():.6f}"
    #                     )

    #                     rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                    
    #                     new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #                     new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                        
    #                     new_c2w = original_pose_backup.clone()
    #                     new_c2w[0, :3, :3] = new_rotation
    #                     new_c2w[0, :3, 3] = new_translation
                        
    #                     camera_secret.camera_to_worlds = new_c2w
    #                     optimized_camera = camera_secret
                        
    #                     final_outputs = self.model(optimized_camera)
    #                     rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
                        
    #                     # Compute final LPIPS score
    #                     model_rgb = (final_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)
    #                     masked_model_rgb = model_rgb * mask_tensor
                        
    #                     save_image(rendered_image, 
    #                             image_dir / f"optimized_secret_view_{self.secret_view_idx}_step_{pose_iter}_loss_{total_loss.item():.6f}.png")
            
    #         # # After optimization, restore model parameter gradient states
    #         # for name, param in self.model.named_parameters():
    #         #     param.requires_grad = model_param_grad_states[name]    

    #         # secret data preparation
    #         with torch.no_grad():
    #             rotation_matrix = self._axis_angle_to_rotation_matrix(self.rotation_offset)
                
    #             new_rotation = rotation_matrix @ original_pose_backup[0, :3, :3]
    #             new_translation = original_pose_backup[0, :3, 3] + self.translation_offset
                
    #             new_c2w = original_pose_backup.clone()
    #             new_c2w[0, :3, :3] = new_rotation
    #             new_c2w[0, :3, 3] = new_translation
                
    #             camera_secret.camera_to_worlds = new_c2w
    #             optimized_camera = camera_secret
                
    #             final_outputs = self.model(optimized_camera)
    #             rendered_image = final_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)  
                
    #             # secret data preparation
    #             self.camera_secret, self.data_secret = optimized_camera, data_secret
    #             self.original_image_secret = rendered_image # [bs, c, h, w]
    #             self.depth_image_secret = final_outputs["depth"].detach().permute(2, 0, 1) # [bs, h, w]
    #             # original secret edges
    #             self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)

    #         torch.cuda.empty_cache()

    #     # start editing
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #     # if (self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss

    #                 # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #                 ref_loss = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #                 loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #             ref_loss = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #             loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
    ######################################################

    # __IGS2GS_IN2N_pie_edge_ref_best__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_ref__best_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if self.first_step:
    #         self.first_step = False
    #         # find the best secret view that align with the reference image
    #         current_secret_idx = 0
    #         current_lpips_score = float("inf")
    #         for idx in tqdm(range(len(self.datamanager.cached_train))):
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)

    #             # compute masked lpips value
    #             mask_np = self.ip2p_ptd.mask
    #             # Convert mask to tensor and ensure it's the right shape/device
    #             mask_tensor = torch.from_numpy(mask_np).float()
    #             if len(mask_tensor.shape) == 2:
    #                 mask_tensor = mask_tensor.unsqueeze(0)  # Add channel dimension
    #             if mask_tensor.shape[0] == 1:
    #                 mask_tensor = mask_tensor.repeat(3, 1, 1)  # Repeat for RGB channels
    #             mask_tensor = mask_tensor.unsqueeze(0)  # Add batch dimension
    #             mask_tensor = mask_tensor.to(self.ref_image_tensor.device)

    #             # Prepare model output
    #             model_rgb = (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1)

    #             # Apply mask to both images
    #             masked_model_rgb = model_rgb * mask_tensor
    #             masked_ref_image = self.ref_image_tensor * mask_tensor

    #             # Compute masked LPIPS score
    #             lpips_score = self.lpips_loss_fn(
    #                 masked_model_rgb,
    #                 masked_ref_image
    #             ).item()

    #             # # unmasked lpips score
    #             # lpips_score = self.lpips_loss_fn(
    #             #     (model_outputs["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #             #     self.ref_image_tensor
    #             # ).item()

    #             if lpips_score < current_lpips_score:
    #                 current_lpips_score = lpips_score
    #                 current_secret_idx = idx

    #         self.secret_view_idx = current_secret_idx
    #         camera, data = self.datamanager.next_train_idx(self.secret_view_idx)
    #         model_outputs = self.model(camera)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image(rendered_image, image_dir / f"best_secret_view_{self.secret_view_idx}_lpips_score_{current_lpips_score}.png")

    #         CONSOLE.print(f"Best secret view index: {self.secret_view_idx} with LPIPS score: {current_lpips_score}")

    #         # secret data preparation
    #         self.camera_secret, self.data_secret = self.datamanager.next_train_idx(self.secret_view_idx)
    #         self.original_image_secret = self.datamanager.original_cached_train[self.secret_view_idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         self.depth_image_secret = self.datamanager.original_cached_train[self.secret_view_idx]["depth"] # [bs, h, w]
    #         # original secret edges
    #         self.original_secret_edges = SobelFilter(ksize=3, use_grayscale=self.config_secret.use_grayscale)(self.original_image_secret)

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss

    #                 # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #                 ref_loss = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #                 loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #             ref_loss = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #             loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        #####################################################

    # __IGS2GS_IN2N_pie_edge_ref_
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_pie_edge_ref_secret_image_cond_{self.config_secret.image_guidance_scale_ip2p_ptd}_async_{self.config_secret.async_ahead_steps}_contrast_{self.ip2p_ptd.contrast}_non_secret_{self.config_secret.image_guidance_scale_ip2p}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
                    
    #                 # edge loss
    #                 rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #                 edge_loss = self.edge_loss_fn(
    #                     rendered_image_secret.to(self.config_secret.device), 
    #                     self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                     self.original_secret_edges.to(self.config_secret.device),
    #                     # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                     image_dir,
    #                     step
    #                 )
    #                 loss_dict["main_loss"] += edge_loss

    #                 # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #                 ref_loss = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #                 loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss
                    
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # ###########################################
    #                 # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone

    #                 edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                     self.text_embeddings_ip2p.to(self.config_secret.device),
    #                     rendered_image_secret.to(self.dtype),
    #                     self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     False, # is depth tensor
    #                     self.depth_image_secret,
    #                     guidance_scale=self.config_secret.guidance_scale,
    #                     image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                     diffusion_steps=self.config_secret.t_dec,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound,
    #                 )

    #                 # edited_image_secret is B, C, H, W in [0, 1]
    #                 edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #                 # Convert edited_image_target to numpy (NEW TARGET)
    #                 edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #                 edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #                 # Convert mask to numpy
    #                 if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                     # It's a PyTorch tensor
    #                     mask_tensor = self.ip2p_ptd.mask
    #                     if mask_tensor.dim() == 4:
    #                         mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                     elif mask_tensor.dim() == 3:
    #                         mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                     else:
    #                         mask_np = mask_tensor.cpu().numpy()
    #                     mask_np = (mask_np * 255).astype(np.uint8)
    #                 elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                     # It's a PIL Image
    #                     mask_np = np.array(self.ip2p_ptd.mask)
    #                     # Convert to grayscale if needed
    #                     if mask_np.ndim == 3:
    #                         mask_np = mask_np[:, :, 0]  # Take first channel
    #                     # Ensure it's uint8
    #                     if mask_np.dtype != np.uint8:
    #                         if mask_np.max() <= 1.0:
    #                             mask_np = (mask_np * 255).astype(np.uint8)
    #                         else:
    #                             mask_np = mask_np.astype(np.uint8)
    #                 else:
    #                     # If it's already numpy, just use it
    #                     mask_np = self.ip2p_ptd.mask

    #                 # Call the original opencv_seamless_clone function with numpy arrays
    #                 result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #                 # Convert the result back to PyTorch tensor format
    #                 # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #                 edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #                 edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #                 edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #                 # ###########################################

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # compute lpips score between ref image and current secret rendering
    #                 lpips_score = self.lpips_loss_fn(
    #                     (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                     self.ref_image_tensor
    #                 )
    #                 # print(f"lpips score: {lpips_score.item():.6f}")

    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
   
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             # edge loss
    #             rendered_image_secret = model_outputs_secret["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # don't detach the output when we want the grad to backpropagate
    #             edge_loss = self.edge_loss_fn(
    #                 rendered_image_secret.to(self.config_secret.device), 
    #                 self.ip2p_ptd.ref_img_tensor.to(self.config_secret.device),
    #                 self.original_secret_edges.to(self.config_secret.device),
    #                 # self.ip2p_ptd.original_edges.to(self.config_secret.device),
    #                 image_dir,
    #                 step
    #             )
    #             loss_dict["main_loss"] += edge_loss

    #             # ref loss, a content loss of ref image added to the original rgb (L1 + lpips loss)
    #             ref_loss = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             ).squeeze() # need to add squeeze to make the ref_loss a scalar instead of a tensor
    #             loss_dict["main_loss"] += self.config_secret.ref_loss_weight * ref_loss

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # ###########################################
    #             # Convert PyTorch tensors to NumPy arrays for opencv_seamless_clone
    #             edited_image_target, depth_tensor_target = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image_secret.to(self.dtype),
    #                 self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 self.depth_image_secret,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             save_image((edited_image_target.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_target_image.png')

    #             # edited_image_secret is B, C, H, W in [0, 1]
    #             edited_image_np = edited_image_secret.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_np = (edited_image_np * 255).astype(np.uint8)

    #             # Convert edited_image_target to numpy (NEW TARGET)
    #             edited_image_target_np = edited_image_target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             edited_image_target_np = (edited_image_target_np * 255).astype(np.uint8)

    #             # Convert mask to numpy
    #             if hasattr(self.ip2p_ptd.mask, 'cpu'):
    #                 # It's a PyTorch tensor
    #                 mask_tensor = self.ip2p_ptd.mask
    #                 if mask_tensor.dim() == 4:
    #                     mask_np = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    #                 elif mask_tensor.dim() == 3:
    #                     mask_np = mask_tensor.squeeze(0).cpu().numpy()
    #                 else:
    #                     mask_np = mask_tensor.cpu().numpy()
    #                 mask_np = (mask_np * 255).astype(np.uint8)
    #             elif hasattr(self.ip2p_ptd.mask, 'save'):
    #                 # It's a PIL Image
    #                 mask_np = np.array(self.ip2p_ptd.mask)
    #                 # Convert to grayscale if needed
    #                 if mask_np.ndim == 3:
    #                     mask_np = mask_np[:, :, 0]  # Take first channel
    #                 # Ensure it's uint8
    #                 if mask_np.dtype != np.uint8:
    #                     if mask_np.max() <= 1.0:
    #                         mask_np = (mask_np * 255).astype(np.uint8)
    #                     else:
    #                         mask_np = mask_np.astype(np.uint8)
    #             else:
    #                 # If it's already numpy, just use it
    #                 mask_np = self.ip2p_ptd.mask

    #             # Call the original opencv_seamless_clone function with numpy arrays
    #             result_np = opencv_seamless_clone(edited_image_np, edited_image_target_np, mask_np)

    #             # Convert the result back to PyTorch tensor format
    #             # From H, W, C in [0, 255] to B, C, H, W in [0, 1]
    #             edited_image_secret = torch.from_numpy(result_np).float() / 255.0
    #             edited_image_secret = edited_image_secret.permute(2, 0, 1).unsqueeze(0)
    #             edited_image_secret = edited_image_secret.to(self.config_secret.device).to(self.dtype)
    #             # ###########################################

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # compute lpips score between ref image and current secret rendering
    #             lpips_score = self.lpips_loss_fn(
    #                 (model_outputs_secret["rgb"].permute(2, 0, 1).unsqueeze(0) * 2 - 1).clamp(-1, 1),
    #                 self.ref_image_tensor
    #             )
    #             # print(f"lpips score: {lpips_score.item():.6f}")

    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_{lpips_score.item():.6f}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_poisson_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        #####################################################

    # comment this function for 1st stage updating, __IGS2GS + IN2N__seva__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N__seva_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # generate ves views for the first iteration
    #     if self.first_iter:
    #         self.first_iter = False
    #         for i, ves_camera in enumerate(self.ves_cameras):
    #             model_outputs_ves = self.model(ves_camera)
    #             rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image_ves).clamp(0, 1), image_dir / f'{step}_ves_image_{i}.png')

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_render.png')
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_list.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_list.png')
    #             save_image((edited_image_secret.to(self.config_secret.device)).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #             # seva results
    #             all_imgs_path = [str(image_dir / f'{step}_secret_image.png')] + [None] * self.num_targets

    #             print(all_imgs_path)

    #             # Create image conditioning.
    #             image_cond = {
    #                 "img": all_imgs_path,
    #                 "input_indices": self.input_indices,
    #                 "prior_indices": self.anchor_indices,
    #             }
    #             # Create camera conditioning.
    #             camera_cond = {
    #                 "c2w": self.seva_c2ws.clone(),
    #                 "K": self.Ks.clone(),
    #                 "input_indices": list(range(self.num_inputs + self.num_targets)),
    #             }

    #             # run_one_scene -> transform_img_and_K modifies VERSION_DICT["H"] and VERSION_DICT["W"] in-place.
    #             video_path_generator = run_one_scene(
    #                 self.task,
    #                 self.VERSION_DICT,  # H, W maybe updated in run_one_scene
    #                 model=self.MODEL,
    #                 ae=self.AE,
    #                 conditioner=self.CONDITIONER,
    #                 denoiser=self.DENOISER,
    #                 image_cond=image_cond,
    #                 camera_cond=camera_cond,
    #                 save_path=image_dir / f'{step}_seva',
    #                 use_traj_prior=True,
    #                 traj_prior_Ks=self.anchor_Ks,
    #                 traj_prior_c2ws=self.anchor_c2ws,
    #                 seed=self.seed,
    #             )
    #             for _ in video_path_generator:
    #                 pass

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################
    
    # comment this function for 1st stage updating, __IGS2GS + IN2N__2_secrets__
    # we can use only IN2N when the number of images in the dataset is small, since IGS2GS + IN2N has longer training time but same results with IN2N in this case.
    # def get_train_loss_dict(self, step: int):        
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_2_secrets_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # generate ves views for the first iteration
    #     if self.first_iter:
    #         self.first_iter = False
    #         for i, ves_camera in enumerate(self.ves_cameras):
    #             model_outputs_ves = self.model(ves_camera)
    #             rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image_ves).clamp(0, 1), image_dir / f'{step}_ves_image_{i}.png')

    #         model_outputs_secret = self.model(self.camera_secret)
    #         rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         save_image((rendered_image_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v * self.config_secret.secret_loss_weight

    #                 # also edit the second secret view
    #                 model_outputs_secret_2 = self.model(self.camera_secret_2)
    #                 metrics_dict_secret_2 = self.model.get_metrics_dict(model_outputs_secret_2, self.data_secret_2)
    #                 loss_dict_secret_2 = self.model.get_loss_dict(model_outputs_secret_2, self.data_secret_2, metrics_dict_secret_2)
    #                 rendered_image_secret_2 = model_outputs_secret_2["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret_2, depth_tensor_secret_2 = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret_2.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret_2.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=1,
    #                     depth=self.depth_image_secret_2,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )
    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret_2.size() != rendered_image_secret_2.size()):
    #                     edited_image_secret_2 = torch.nn.functional.interpolate(edited_image_secret_2, size=rendered_image_secret_2.size()[2:], mode='bilinear')
                    
    #                 # write edited image to dataloader
    #                 edited_image_secret_2 = edited_image_secret_2.to(self.original_image_secret_2.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx_2]["image"] = edited_image_secret_2.squeeze().permute(1,2,0)
    #                 self.data_secret_2["image"] = edited_image_secret_2.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret_2.items():
    #                     metrics_dict[f"secret_2_{k}"] = v
    #                 for k, v in loss_dict_secret_2.items():
    #                     loss_dict[f"secret_2_{k}"] = v * self.config_secret.secret_loss_weight
                    
    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
    #                     image_save_secret_2 = torch.cat([depth_tensor_secret_2, rendered_image_secret_2, edited_image_secret_2.to(self.config_secret.device), self.original_image_secret_2.to(self.config_secret.device)])
    #                     save_image((image_save_secret_2).clamp(0, 1), image_dir / f'{step}_secret_image_2.png')
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v * self.config_secret.secret_loss_weight

    #             # also update the second secret view w/o editing (important)
    #             model_outputs_secret_2 = self.model(self.camera_secret_2)
    #             metrics_dict_secret_2 = self.model.get_metrics_dict(model_outputs_secret_2, self.data_secret_2)
    #             loss_dict_secret_2 = self.model.get_loss_dict(model_outputs_secret_2, self.data_secret_2, metrics_dict_secret_2)

    #             for k, v in metrics_dict_secret_2.items():
    #                 metrics_dict[f"secret_2_{k}"] = v
    #             for k, v in loss_dict_secret_2.items():
    #                 loss_dict[f"secret_2_{k}"] = v * self.config_secret.secret_loss_weight

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx and idx != self.config_secret.secret_view_idx_2:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         if idx == self.config_secret.secret_view_idx_2:
    #             model_outputs_secret_2 = self.model(self.camera_secret_2)
    #             metrics_dict_secret_2 = self.model.get_metrics_dict(model_outputs_secret_2, self.data_secret_2)
    #             loss_dict_secret_2 = self.model.get_loss_dict(model_outputs_secret_2, self.data_secret_2, metrics_dict_secret_2)
    #             rendered_image_secret_2 = model_outputs_secret_2["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret_2, depth_tensor_secret_2 = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret_2.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret_2.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=1,
    #                 depth=self.depth_image_secret_2,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )
    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret_2.size() != rendered_image_secret_2.size()):
    #                 edited_image_secret_2 = torch.nn.functional.interpolate(edited_image_secret_2, size=rendered_image_secret_2.size()[2:], mode='bilinear')
                
    #             # write edited image to dataloader
    #             edited_image_secret_2 = edited_image_secret_2.to(self.original_image_secret_2.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx_2]["image"] = edited_image_secret_2.squeeze().permute(1,2,0)
    #             self.data_secret_2["image"] = edited_image_secret_2.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret_2.items():
    #                 metrics_dict[f"secret_2_{k}"] = v
    #             for k, v in loss_dict_secret_2.items():
    #                 loss_dict[f"secret_2_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret_2, rendered_image_secret_2, edited_image_secret_2.to(self.config_secret.device), self.original_image_secret_2.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image_2.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################

    # comment this function for 1st stage updating, __IGS2GS + IN2N__
    # we can use only IN2N when the number of images in the dataset is small, since IGS2GS + IN2N has longer training time but same results with IN2N in this case.
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_IN2N_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # generate ves views for the first iteration
    #     # if self.first_iter:
    #     #     self.first_iter = False
    #     #     for i, ves_camera in enumerate(self.ves_cameras):
    #     #         model_outputs_ves = self.model(ves_camera)
    #     #         rendered_image_ves = model_outputs_ves["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #     #         save_image((rendered_image_ves).clamp(0, 1), image_dir / f'{step}_ves_image_{i}.png')

    #     #     model_outputs_secret = self.model(self.camera_secret)
    #     #     rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #     #     save_image((rendered_image_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0 and (self.first_SequentialEdit): # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # the implementation of randomly selecting an index to edit instead of update all images at once
    #         # generate the indexes for non-secret view editing
    #         all_indices = np.arange(len(self.datamanager.cached_train))
    #         allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #         if step % self.config_secret.edit_rate == 0:
    #             #----------------non-secret view editing----------------
    #             # randomly select an index to edit
    #             idx = random.choice(allowed)
    #             camera, data = self.datamanager.next_train_idx(idx)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #             original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #             depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 50 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             #----------------secret view editing----------------
    #             if step % self.config_secret.secret_edit_rate == 0:
    #                 model_outputs_secret = self.model(self.camera_secret)
    #                 metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #                 loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #                 rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #                 edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                     image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                     image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                     secret_idx=0,
    #                     depth=self.depth_image_secret,
    #                     lower_bound=self.config_secret.lower_bound,
    #                     upper_bound=self.config_secret.upper_bound
    #                 )

    #                 # resize to original image size (often not necessary)
    #                 if (edited_image_secret.size() != rendered_image_secret.size()):
    #                     edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #                 # write edited image to dataloader
    #                 edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #                 self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #                 self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #                 for k, v in metrics_dict_secret.items():
    #                     metrics_dict[f"secret_{k}"] = v
    #                 for k, v in loss_dict_secret.items():
    #                     loss_dict[f"secret_{k}"] = v
                    
    #                 # save edited secret image
    #                 if step % 50 == 0:
    #                     image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                     save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
    #         else:
    #             # non-editing steps loss computing
    #             camera, data = self.datamanager.next_train(step)
    #             model_outputs = self.model(camera)
    #             metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #             loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #             # also update the secret view w/o editing (important)
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 secret_idx=0,
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False
    #             self.first_SequentialEdit = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # comment this function for 1st stage updating, __IN2N__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IN2N_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)

    #     # print(step) # start from 30000

    #     # the implementation of randomly selecting an index to edit instead of update all images at once
    #     # generate the indexes for non-secret view editing
    #     all_indices = np.arange(len(self.datamanager.cached_train))
    #     allowed = all_indices[all_indices != self.config_secret.secret_view_idx]

    #     if step % self.config_secret.edit_rate == 0:
    #         #----------------non-secret view editing----------------
    #         # randomly select an index to edit
    #         idx = random.choice(allowed)
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #         edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #             self.text_embeddings_ip2p.to(self.config_secret.device),
    #             rendered_image.to(self.dtype),
    #             original_image.to(self.config_secret.device).to(self.dtype),
    #             False, # is depth tensor
    #             depth_image,
    #             guidance_scale=self.config_secret.guidance_scale,
    #             image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #             diffusion_steps=self.config_secret.t_dec,
    #             lower_bound=self.config_secret.lower_bound,
    #             upper_bound=self.config_secret.upper_bound,
    #         )

    #         # resize to original image size (often not necessary)
    #         if (edited_image.size() != rendered_image.size()):
    #             edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #         # write edited image to dataloader
    #         edited_image = edited_image.to(original_image.dtype)
    #         self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #         data["image"] = edited_image.squeeze().permute(1,2,0)

    #         # save edited non-secret image
    #         if step % 50 == 0:
    #             image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #             save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         #----------------secret view editing----------------
    #         if step % self.config_secret.secret_edit_rate == 0:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             if step % 50 == 0:
    #                 image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #                 save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')
    #                 save_image(edited_image_secret.to(self.config_secret.device).clamp(0, 1), image_dir / f'{step}_secret_image_seva.png')
    #     else:
    #         # non-editing steps loss computing
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # also update the secret view w/o editing (important)
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################


    # comment this function for 1st stage updating, __IGS2GS__
    # def get_train_loss_dict(self, step: int):
    #     ######################################################
    #     # Secret view updating                              
    #     ######################################################
    #     base_dir = self.trainer.base_dir
    #     image_dir = base_dir / f"images_IGS2GS_secret_{self.config_secret.image_guidance_scale_ip2p_ptd}_{self.config_secret.secret_edit_rate}_non_secret_{self.config_secret.image_guidance_scale_ip2p}_{self.config_secret.edit_rate}"
    #     if not image_dir.exists():
    #         image_dir.mkdir(parents=True, exist_ok=True)
    
    #     # if ((step-1) % self.config.gs_steps) == 0:
    #     if (step % self.config.gs_steps) == 0: # update also for the first step
    #         self.makeSequentialEdits = True

    #     if (not self.makeSequentialEdits):
    #         # update the non-secret views w/o editing
    #         camera, data = self.datamanager.next_train(step)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)
    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         if step % 500 == 0:
    #             rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             save_image((rendered_image).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #         # also update the secret view w/o editing
    #         model_outputs_secret = self.model(self.camera_secret)
    #         metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #         loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)

    #         for k, v in metrics_dict_secret.items():
    #             metrics_dict[f"secret_{k}"] = v
    #         for k, v in loss_dict_secret.items():
    #             loss_dict[f"secret_{k}"] = v

    #         if step % 500 == 0:
    #             # save the secret view image
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             image_save_secret = torch.cat([rendered_image_secret, self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #     else:
    #         # get index
    #         idx = self.curr_edit_idx
    #         camera, data = self.datamanager.next_train_idx(idx)
    #         model_outputs = self.model(camera)
    #         metrics_dict = self.model.get_metrics_dict(model_outputs, data)

    #         original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
    #         rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

    #         depth_image = self.datamanager.original_cached_train[idx]["depth"] # [bs, h, w]

    #         # edited_image = self.ip2p.edit_image(
    #         #             self.text_embedding.to(self.ip2p_device),
    #         #             rendered_image.to(self.ip2p_device),
    #         #             original_image.to(self.ip2p_device),
    #         #             guidance_scale=self.config.guidance_scale,
    #         #             image_guidance_scale=self.config.image_guidance_scale,
    #         #             diffusion_steps=self.config.diffusion_steps,
    #         #             lower_bound=self.config.lower_bound,
    #         #             upper_bound=self.config.upper_bound,
    #         #         )
            
    #         # edit image using IP2P depth when idx != secret_view_idx
    #         if idx != self.config_secret.secret_view_idx:
    #             edited_image, depth_tensor = self.ip2p_depth.edit_image_depth(
    #                 self.text_embeddings_ip2p.to(self.config_secret.device),
    #                 rendered_image.to(self.dtype),
    #                 original_image.to(self.config_secret.device).to(self.dtype),
    #                 False, # is depth tensor
    #                 depth_image,
    #                 guidance_scale=self.config_secret.guidance_scale,
    #                 image_guidance_scale=self.config_secret.image_guidance_scale_ip2p,
    #                 diffusion_steps=self.config_secret.t_dec,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound,
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image.size() != rendered_image.size()):
    #                 edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image = edited_image.to(original_image.dtype)
    #             self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
    #             data["image"] = edited_image.squeeze().permute(1,2,0)

    #             # save edited non-secret image
    #             if step % 25 == 0:
    #                 image_save_non_secret = torch.cat([depth_tensor, rendered_image, edited_image.to(self.config_secret.device), original_image.to(self.config_secret.device)])
    #                 save_image((image_save_non_secret).clamp(0, 1), image_dir / f'{step}_non_secret_image.png')

    #         loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)

    #         # edit image using IP2P + PTD when idx == secret_view_idx
    #         if idx == self.config_secret.secret_view_idx:
    #             model_outputs_secret = self.model(self.camera_secret)
    #             metrics_dict_secret = self.model.get_metrics_dict(model_outputs_secret, self.data_secret)
    #             loss_dict_secret = self.model.get_loss_dict(model_outputs_secret, self.data_secret, metrics_dict_secret)
    #             rendered_image_secret = model_outputs_secret["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)
    #             edited_image_secret, depth_tensor_secret = self.ip2p_ptd.edit_image_depth(
    #                 image=rendered_image_secret.to(self.dtype), # input should be B, 3, H, W, in [0, 1]
    #                 image_cond=self.original_image_secret.to(self.config_secret.device).to(self.dtype),
    #                 depth=self.depth_image_secret,
    #                 lower_bound=self.config_secret.lower_bound,
    #                 upper_bound=self.config_secret.upper_bound
    #             )

    #             # resize to original image size (often not necessary)
    #             if (edited_image_secret.size() != rendered_image_secret.size()):
    #                 edited_image_secret = torch.nn.functional.interpolate(edited_image_secret, size=rendered_image_secret.size()[2:], mode='bilinear')

    #             # write edited image to dataloader
    #             edited_image_secret = edited_image_secret.to(self.original_image_secret.dtype)
    #             self.datamanager.cached_train[self.config_secret.secret_view_idx]["image"] = edited_image_secret.squeeze().permute(1,2,0)
    #             self.data_secret["image"] = edited_image_secret.squeeze().permute(1,2,0)

    #             for k, v in metrics_dict_secret.items():
    #                 metrics_dict[f"secret_{k}"] = v
    #             for k, v in loss_dict_secret.items():
    #                 loss_dict[f"secret_{k}"] = v
                
    #             # save edited secret image
    #             image_save_secret = torch.cat([depth_tensor_secret, rendered_image_secret, edited_image_secret.to(self.config_secret.device), self.original_image_secret.to(self.config_secret.device)])
    #             save_image((image_save_secret).clamp(0, 1), image_dir / f'{step}_secret_image.png')

    #         #increment curr edit idx
    #         # and update all the images in the dataset
    #         self.curr_edit_idx += 1
    #         # self.makeSequentialEdits = False
    #         if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
    #             self.curr_edit_idx = 0
    #             self.makeSequentialEdits = False

    #     return model_outputs, loss_dict, metrics_dict
        ######################################################