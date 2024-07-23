import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2

        # TODO: Define model
        self.wnll1 = nn.utils.weight_norm(nn.Linear(latent_size+3,512))
        self.wnll2 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll3 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll4 = nn.utils.weight_norm(nn.Linear(512,512-latent_size-3))

        self.wnll5 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll6 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll7 = nn.utils.weight_norm(nn.Linear(512,512))
        self.wnll8 = nn.utils.weight_norm(nn.Linear(512,512))

        self.fc = nn.Linear(512,1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        # TODO: implement forward pass
        x = self.dropout(self.relu(self.wnll1(x_in)))
        x = self.dropout(self.relu(self.wnll2(x)))
        x = self.dropout(self.relu(self.wnll3(x)))
        x = self.dropout(self.relu(self.wnll4(x)))
        
        x = torch.cat((x,x_in),dim=1)
        
        x = self.dropout(self.relu(self.wnll5(x)))
        x = self.dropout(self.relu(self.wnll6(x)))
        x = self.dropout(self.relu(self.wnll7(x)))
        x = self.dropout(self.relu(self.wnll8(x)))
        
        x = self.fc(x)
        return x
