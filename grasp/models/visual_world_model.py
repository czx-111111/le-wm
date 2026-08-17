import torch
import torch.nn as nn
from torchvision import transforms

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat

        # Handle DDP-wrapped encoders by accessing .module attribute
        encoder_unwrapped = encoder.module if hasattr(encoder, 'module') else encoder
        self.emb_dim = encoder_unwrapped.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        # Handle DDP-wrapped encoders by accessing .module attribute
        encoder_unwrapped = encoder.module if hasattr(encoder, 'module') else encoder
        if "dino" in encoder_unwrapped.name or "vjepa" in encoder_unwrapped.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder_unwrapped.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act):
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)

        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            f = z_dct['visual'].shape[2]
            proprio_tiled = z_dct['proprio'].unsqueeze(2).expand(-1, -1, f, -1)
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = act_emb.unsqueeze(2).expand(-1, -1, f, -1)
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)

        return z

    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act

    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b, t = visual.shape[0], visual.shape[1]
        visual = visual.reshape(b * t, *visual.shape[2:])
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        p, d = visual_embs.shape[1], visual_embs.shape[2]
        visual_embs = visual_embs.reshape(b, t, p, d)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        b, T, p, d = z.shape
        # reshape to a batch of windows of inputs
        z = z.reshape(b, T * p, d)
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = z.reshape(b, T, p, d)
        return z

    def decode(self, z, include_actions=True):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z, include_actions)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def visualize(self, z):
        """
        Visualize encoded states by decoding them. Adheres to more
        general framework for planners

        Args:
            z: Encoded states (b, num_frames, num_patches, emb_dim)

        Returns:
            obs: Decoded observations
        """
        return self.decode(z, include_actions=False)[0]["visual"]

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        c, h, w = visual.shape[1], visual.shape[2], visual.shape[3]
        visual = visual.reshape(b, num_frames, c, h, w)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff


    def separate_emb(self, z, include_actions=True):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """


        if self.concat_dim == 0:
            if include_actions:
                z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
            else:
                z_visual, z_proprio, z_act = z[:, :, :-1, :], z[:, :, -1, :], None

        elif self.concat_dim == 1:
            if include_actions:
                z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                            z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                            z[..., -self.action_dim:]
            else:
                z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim)], \
                                            z[..., -(self.proprio_dim) :],  \
                                            None
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]

            if include_actions:
                z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]

        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def combine_emb(self, z_obs, z_act=None):
        """
        input: z_obs (dict): {"visual": visual_emb, "proprio": proprio_emb}, z_act (tensor)
        output: z (tensor): combined embedding

        This method is the exact inverse of separate_emb. Can handle no action, or when z_obs is not a dict
        (assumed to be encoded concatenated tensor in this case)
        """
        if self.concat_dim == 0:
            # Get the number of patches from visual embedding
            num_patches = z_obs['visual'].shape[2]
            # Repeat proprio and action embeddings to match visual's patch dimension
            proprio_expanded = z_obs['proprio'].unsqueeze(2)
            if z_act is not None:
                act_expanded = z_act.unsqueeze(2)
                z = torch.cat(
                    [z_obs['visual'], proprio_expanded, act_expanded],
                    dim=2  # concatenate as tokens
                )
            else:
                z = torch.cat(
                    [z_obs['visual'], proprio_expanded],
                    dim=2  # concatenate as tokens
                )
        elif self.concat_dim == 1:
            if isinstance(z_obs, dict):
                # Get the number of patches from visual embedding
                num_patches = z_obs['visual'].shape[2]
                # Repeat proprio and action embeddings to match visual's patch dimension
                proprio_expanded = z_obs['proprio'].unsqueeze(2).repeat(1, 1, num_patches, self.num_proprio_repeat)
                if z_act is not None:
                    act_expanded = z_act.unsqueeze(2).repeat(1, 1, num_patches, self.num_action_repeat)
                    z = torch.cat(
                        [z_obs['visual'], proprio_expanded, act_expanded],
                        dim=-1  # concatenate as tokens
                    )
                else:
                    z = torch.cat(
                        [z_obs['visual'], proprio_expanded],
                        dim=-1  # concatenate as tokens
                    )
            else: # assumed to already be combined tensor
                # act must not be None
                num_patches = z_obs.shape[2]
                act_expanded = z_act.unsqueeze(2).repeat(1, 1, num_patches, self.num_action_repeat)
                z = torch.cat(
                    [z_obs, act_expanded],
                    dim=-1  # concatenate as tokens
                )
        return z


    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim],
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            f = z.shape[2]
            act_tiled = act_emb.unsqueeze(2).expand(-1, -1, f, -1)
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z = self.simulate_one_step(action, z, t, inc)
            t += inc


        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

    def rollout_noisy(self, obs_0, act, var_states):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
                  var_states: scalar quantity for scaling Gaussian noise
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z = self.simulate_one_step_noisy(action, z, t, inc, var_states)
            t += inc


        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z

    def simulate_one_step_noisy(self, action, z, t, inc, var_states):
        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -inc:, ...]
        z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
        # Add Gaussian noise to state embeddings (excluding actions)
        if var_states > 0:
            state_noise = torch.randn_like(z_new[..., :-self.action_dim], device=z_new.device) * var_states
            z_new[..., :-self.action_dim] += state_noise
        z = torch.cat([z, z_new], dim=1)
        return z


    def simulate_one_step(self, action, z, t, inc):
        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -inc:, ...]
        z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
        z = torch.cat([z, z_new], dim=1)
        return z

    def encode_observation_to_vector(self, obs):
        """
        Encode initial and goal observations into the combined embedding space.

        Args:
            obs_0: Initial observations dict with 'visual' and 'proprio' keys
            obs_g: Goal observations dict with 'visual' and 'proprio' keys

        Returns:
            z_obs_0: Encoded initial observation embedding with grad enabled
            z_obs_g: Encoded goal observation embedding with grad enabled
        """
        # Encode obs_g
        output_dict = self.encode_obs(obs)
        obs_dict = {k: v.detach() for k, v in output_dict.items()}
        z_obs = self.combine_emb(obs_dict)
        z_obs.requires_grad = True

        return z_obs

    def next_state(self, z_state_expanded, action_samples_flat):
        """
        Batched next state prediction for efficient planning.

        Args:
            z_state_expanded: Expanded state sequence [expanded_batch, horizon+1, patch_dim, z_dim]
            action_samples_flat: Flattened action samples [expanded_batch, horizon, action_dim]
            horizon: Horizon length

        Returns:
            f_next_state_list: Predicted next states [expanded_batch, horizon, patch_dim, z_dim]
        """
        expanded_n_evals = z_state_expanded.shape[0]
        horizon = action_samples_flat.shape[1]
        device = z_state_expanded.device

        # Encode all action samples
        z_actions = self.action_encoder(action_samples_flat)
        z_state_actions = self.combine_emb(z_state_expanded[:, :-1], z_actions)

        if horizon <= self.num_hist:
            next_state_action_full = self.predict(z_state_actions)
        else:
            z_state_actions_unrolled = z_state_actions.unfold(1, self.num_hist, 1)
            # result after permute is a tensor of shape (B, (horizon+1-num_hist), num_hist, patch_dim, z_dim)
            z_state_actions_unrolled = z_state_actions_unrolled.permute(0, 1, 4, 2, 3)
            # flatten into batch
            b, k = z_state_actions_unrolled.shape[0], z_state_actions_unrolled.shape[1]
            z_state_actions_unrolled = z_state_actions_unrolled.reshape(b * k, *z_state_actions_unrolled.shape[2:])

            n_sa_rollbatched = self.predict(z_state_actions_unrolled)
            # first unflatten again
            n_sa_full_unroll = n_sa_rollbatched.reshape(expanded_n_evals, k, *n_sa_rollbatched.shape[1:])

            next_state_action_full = torch.zeros(
                expanded_n_evals, horizon, *z_state_actions.shape[2:], device=device
            )
            # get the first window
            next_state_action_full[:, :self.num_hist] = n_sa_full_unroll[:, 0, ...]
            # get the last element of every following window
            next_state_action_full[:, self.num_hist:] = n_sa_full_unroll[:, 1:, -1, ...]

        f_next_state_list = next_state_action_full[..., : -self.action_dim]
        return f_next_state_list

    def rollout_encoded(self, z_obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """

        # encoded action for init z, unencoded actions
        # for the rest
        z_actions = self.action_encoder(act)
        act_0 = z_actions[:, :1]
        action = act[:, 1:]

        z = self.combine_emb(z_obs_0, act_0)

        # Pre-compute iteration count for torch.compile efficiency
        num_steps = action.shape[1]
        for t in range(num_steps):
            z = self.simulate_one_step(action, z, t, 1)

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z
