import random
from pathlib import Path

import torch

from ..data.objaverse import Objaverse
from ..model.deepsdf import DeepSDFDecoder
from ..util.misc import evaluate_model_on_grid

class InferenceHandlerDeepSDF:

    def __init__(self, latent_code_length, color_latent_code_length, one_hot_encoding_length, experiment, device):
        """
        :param latent_code_length: latent code length for the trained DeepSDF model
        :param experiment: path to experiment folder for the trained model; should contain "model_best.ckpt" and "latent_best.ckpt"
        :param device: torch device where inference is run
        """
        self.latent_code_length = latent_code_length
        self.color_latent_code_length = color_latent_code_length
        self.one_hot_encoding_length = one_hot_encoding_length
        self.experiment = Path(experiment)
        self.device = device
        self.truncation_distance = 0.01
        self.num_samples = 4096

    def get_model(self):
        """
        :return: trained deep sdf model loaded from disk
        """
        model = DeepSDFDecoder(self.latent_code_length, self.color_latent_code_length, self.one_hot_encoding_length)
        model.load_state_dict(torch.load(self.experiment / "model_best.ckpt", map_location='cpu'))
        model.eval()
        model.to(self.device)
        return model

    def get_latent_codes(self):
        """
        :return: latent codes which were optimized during training
        """
        latent_codes = torch.nn.Embedding.from_pretrained(torch.load(self.experiment / "latent_best.ckpt", map_location='cpu')['weight'])
        colour_latent_codes = torch.nn.Embedding.from_pretrained(torch.load(self.experiment / "color_latent_best.ckpt", map_location='cpu')['weight'])
        latent_codes.to(self.device)
        colour_latent_codes.to(self.device)
        return latent_codes, colour_latent_codes

    def reconstruct(self, points, sdf, colours, viewing_dirs, class_label, num_optimization_iters):
        """
        Reconstructs by optimizing a latent code that best represents the input sdf observations
        :param points: all observed points for the shape which needs to be reconstructed
        :param sdf: all observed sdf values corresponding to the points
        :param num_optimization_iters: optimization is performed for this many number of iterations
        :return: tuple with mesh representations of the reconstruction
        """

        model = self.get_model()

        # define loss criterion for optimization
        loss_l1 = torch.nn.L1Loss()

        # initialize the latent vector that will be optimized
        latent = torch.ones(1, self.latent_code_length).normal_(mean=0, std=0.01).to(self.device)
        latent_col = torch.ones(1, self.color_latent_code_length).normal_(mean=0, std=0.01).to(self.device)
        latent.requires_grad = True
        latent_col.requires_grad = True

        # create optimizer on latent, use a learning rate of 0.005
        optimizer = torch.optim.Adam([latent, latent_col], lr=0.005)

        for iter_idx in range(num_optimization_iters):
            # zero out gradients
            optimizer.zero_grad()

            # sample a random batch from the observations, batch size = self.num_samples
            batch_indices = random.sample(range(0, len(points)), self.num_samples)

            batch_points = points[batch_indices, :]
            batch_sdf = sdf[batch_indices, :]
            batch_colours = colours[batch_indices, :]
            batch_viewing_dirs = viewing_dirs[batch_indices, :]
            
            # move batch to device
            batch_points = batch_points.to(self.device)
            batch_sdf = batch_sdf.to(self.device)
            batch_colours = batch_colours.to(self.device)
            batch_viewing_dirs = batch_viewing_dirs.to(self.device)

            class_labels = torch.tensor(Objaverse.get_class_id(class_label)).to(self.device)
            class_label_one_hot = torch.nn.functional.one_hot(class_labels, num_classes=4).float().unsqueeze(0).expand(self.num_samples, -1)

            # same latent code is used per point, therefore expand it to have same length as batch points
            latent_codes = latent.expand(self.num_samples, -1)
            latent_col_codes = latent_col.expand(self.num_samples, -1)

            # forward pass with latent_codes and batch_points
            predicted_sdf, predicted_colours = model(batch_points, batch_viewing_dirs, latent_codes, latent_col_codes, class_label_one_hot)

            # truncate predicted sdf between -0.1, 0.1
            predicted_sdf = predicted_sdf.clamp(-0.1, 0.1)

            # compute loss wrt to observed sdf
            shape_loss = loss_l1(predicted_sdf, batch_sdf)
            colour_loss = loss_l1(predicted_colours, batch_colours)

            # regularize latent code
            shape_loss += 1e-4 * torch.mean(latent.pow(2))
            colour_loss += 1e-4 * torch.mean(latent_col.pow(2))

            # combine shape and colour loss
            loss = shape_loss + colour_loss

            # backwards and step
            loss.backward()
            optimizer.step()

            # loss logging
            if iter_idx % 50 == 0:
                print(f'[{iter_idx:05d}] optim_loss: {loss.cpu().item():.6f}')

        print('Optimization complete.')

        # visualize the reconstructed shape
        vertices, faces, vertex_colours = evaluate_model_on_grid(model, latent.squeeze(0), latent_col.squeeze(0), self.device, 64, None)
        return vertices, faces, vertex_colours

    def interpolate(self, shape_0_id, shape_1_id, num_interpolation_steps):
        """
        Interpolates latent codes between provided shapes and exports the intermediate reconstructions
        :param shape_0_id: first shape identifier
        :param shape_1_id: second shape identifier
        :param num_interpolation_steps: number of intermediate interpolated points
        :return: None, saves the interpolated shapes to disk
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes, train_latent_col_codes = self.get_latent_codes()

        # get indices of shape_ids latent codes
        train_items = Objaverse(4096, "train").items
        latent_code_indices = torch.LongTensor([train_items.index(shape_0_id), train_items.index(shape_1_id)]).to(self.device)

        # get latent codes for provided shape ids
        latent_codes = train_latent_codes(latent_code_indices)
        col_latent_codes = train_latent_col_codes(latent_code_indices)

        for i in range(0, num_interpolation_steps + 1):
            # interpolate the latent codes: latent_codes[0, :] and latent_codes[1, :]
            w = i / num_interpolation_steps
            interpolated_code = (1-w) * latent_codes[0, :] + w * latent_codes[1, :]
            interpolated_color_code = (1-w) * col_latent_codes[0, :] + w * col_latent_codes[1, :]
            # reconstruct the shape at the interpolated latent code
            evaluate_model_on_grid(model, interpolated_code, interpolated_color_code, self.device, 64, self.experiment / "interpolation" / f"{i:05d}_000.obj")

    def infer_from_latent_code(self, latent_code_index):
        """
        Reconstruct shape from a given latent code index
        :param latent_code_index: shape index for a shape in the train set for which reconstruction is performed
        :return: tuple with mesh representations of the reconstruction
        """

        # get saved model and latent codes
        model = self.get_model()
        train_latent_codes, train_latent_col_codes = self.get_latent_codes()

        # get latent code at given index
        latent_code_indices = torch.LongTensor([latent_code_index]).to(self.device)
        latent_codes = train_latent_codes(latent_code_indices)

        # get latent color code at given index
        col_latent_codes = train_latent_col_codes(latent_code_indices)

        # reconstruct the shape at latent code
        vertices, faces = evaluate_model_on_grid(model, latent_codes[0], col_latent_codes[0], self.device, 64, None)

        return vertices, faces

