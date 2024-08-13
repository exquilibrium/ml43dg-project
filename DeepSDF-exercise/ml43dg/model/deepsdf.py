import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, shape_latent_size, colour_latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        self.wnll1 = nn.utils.weight_norm(nn.Linear(shape_latent_size+3, 512))
        self.wnll2 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll3 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll4 = nn.utils.weight_norm(nn.Linear(512,512-shape_latent_size-3))

        self.wnll5 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll6 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll7 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll8 = nn.utils.weight_norm(nn.Linear(512,512))

        self.wnll5_col = nn.utils.weight_norm(nn.Linear(512+colour_latent_size+2, 512))

        self.fc = nn.Linear(512,1)
        # Sigmoid to ensure output is in [0,1]
        self.fc_col = nn.Linear(512,3)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, point, viewing_direction, shape_latent_code, colour_latent_code):
        """
        :param point: B x 3 tensor of point coordinates
        :param viewing_direction: B x 2 tensor of viewing direction
        :param latent_code: B x latent_size tensor of latent codes
        :return: B x 1 tensor
        """
        x_in = torch.cat([point, shape_latent_code], dim=1)
        x = self.dropout(self.relu(self.wnll1(x_in)))
        x = self.dropout(self.relu(self.wnll2(x)))
        x = self.dropout(self.relu(self.wnll3(x)))
        x = self.dropout(self.relu(self.wnll4(x)))
        
        shared_features = torch.cat((x, x_in),dim=1)
        
        # SDF prediction
        sdf_features = self.dropout(self.relu(self.wnll5(shared_features)))
        sdf_features = self.dropout(self.relu(self.wnll6(sdf_features)))
        sdf_features = self.dropout(self.relu(self.wnll7(sdf_features)))
        sdf_features = self.dropout(self.relu(self.wnll8(sdf_features)))
        sdf_output = self.fc(sdf_features)

        # Colour prediction
        colour_in = torch.cat([shared_features, colour_latent_code, viewing_direction], dim=1)
        colour_features = self.dropout(self.relu(self.wnll5_col(colour_in)))
        colour_features = self.dropout(self.relu(self.wnll6(colour_features)))
        colour_features = self.dropout(self.relu(self.wnll7(colour_features)))
        colour_features = self.dropout(self.relu(self.wnll8(colour_features)))
        colour_output = self.sigmoid(self.fc_col(colour_features))

        return sdf_output, colour_output
