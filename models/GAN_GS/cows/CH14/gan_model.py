import os
import sys
import time 
import datetime
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

# Get the absolute path to the metric directory
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../../../"))
metric_dir = os.path.join(project_root, "metric")
# Add metric directory to sys.path
if metric_dir not in sys.path:
    sys.path.append(metric_dir)
# Now import the evaluator
from evaluator import train_evaluator

# The Residual Blocks for Generator and Discriminator
class ResNetBlockGenerator(nn.Module):
    def __init__(self, input_dim, alpha = 0.05):
        super(ResNetBlockGenerator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),  
            nn.LeakyReLU(alpha),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim)  
        )
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        residual = x
        x = self.block(x)
        return self.leaky_relu(x + residual)

class ResNetBlockDiscriminator(nn.Module):
    def __init__(self, input_dim, alpha=0.05):
        super(ResNetBlockDiscriminator, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(alpha),
            nn.Linear(input_dim, input_dim), 
        )
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        residual = x
        x = self.block(x)
        return self.leaky_relu(x + residual)

# Generator
class ResGenerator(nn.Module):
    """
    Generator class
    """
    def __init__(self, 
                 latent_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 output_dim: int,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()

        """Parameters:
            latent_dim (int): dimension of latent noise vector z
            hidden_dim1 (int): dimension of 1st hidden layer
            hidden_dim2 (int): dimension of 2nd hidden layer
            output_dim (int): dimension of generated data (nb_SNPs)
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """
        # Dimensions
        self.latent_dim = latent_dim
        # Layers params
        self.output_dim = output_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dims=[hidden_dim1, hidden_dim2]

        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Initial Layer (Before ResNet Block)
        self.initial = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),  
            self.activation_func
        )

        # (n_hidden_dim - 1) ResNet block
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.res_blocks.append(nn.Sequential(
                ResNetBlockGenerator(self.hidden_dims[i], self.negative_slope),
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),  # Transition layer
                nn.LeakyReLU(self.negative_slope) 
            ))

        #final layer
        self.final = nn.Linear(self.hidden_dims[-1], 3*self.output_dim)
            

    def forward(self, z: torch.tensor, temperature = 1.0, hard = False):
        """ Main function to generate from input noise vector.
        ----
        Parameters:
            x (torch.tensor): input noise vector
            temperature (float): The temperature parameter (τ) in Gumbel-Softmax controls the sharpness of the output distribution
            hard (bool): The hard parameter in Gumbel-Softmax determines whether the output is a soft probability distribution (hard=False during training)
            or a one-hot discrete vector (hard=True during inference)
        Returns:
            (torch.tensor): generated data
        """
        x = self.initial(z)
        for block in self.res_blocks:
            x = block(x)
        x = self.final(x)
        logits = x.view(-1, self.output_dim, 3)  # Reshape to [BATCH_SIZE, seq_length, 3]
        output = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        return output

# Discriminator
class ResDiscriminator(nn.Module):
    """ Discriminator """
    def __init__(self, x_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()
        """Parameters:
            x_dim (int): dimension of SNP sequence length
            hidden_dim1 (int): dimension of 1st hidden layer.
            hidden_dim2 (int): dimension of 2nd hidden layer.
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """
        # Layers params
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dims=[hidden_dim1, hidden_dim2]
        
        # Dimensions
        self.x_dim = x_dim
        
        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Initial Layer
        self.initial = nn.Sequential(nn.Linear(self.x_dim *3, self.hidden_dims[0]), nn.LeakyReLU(self.negative_slope))

        # ResBlock Layer
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.res_blocks.append(nn.Sequential(
                ResNetBlockDiscriminator(self.hidden_dims[i], self.negative_slope),
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),  # Transition layer
                nn.LeakyReLU(self.negative_slope) 
            ))
            
        # Final Layer
        self.final = nn.Linear(self.hidden_dims[-1], 1)
        self.final_activation = nn.Sigmoid() # Vanilla GAN is a binary classification problem

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten Input
        x = self.initial(x)
        for block in self.res_blocks:
            x = block(x)
        return self.final_activation(self.final(x)) 

# GAN model
class GAN(object):
    """
    GAN
    """
    def __init__(self, config: dict, device: torch.device):
        """ Parameters:
            config (dict): model architecture dictionary
            device: which CPU or GPU(0,1,2) to use
        """        
        # Set Architecture
        self.latent_dim = config['latent_dim']
        self.x_dim = config['x_dim']
        self.hidden_dim1_g = config['hidden_dim1_g']
        self.hidden_dim2_g = config['hidden_dim2_g']
        self.hidden_dim1_d = config['hidden_dim1_d']
        self.hidden_dim2_d = config['hidden_dim2_d']
        self.lr_g = config['lr_g']
        self.lr_d = config['lr_d']

        # Set device
        self.device = device

        # Set Generator
        self.G = ResGenerator(
            self.latent_dim,
            self.hidden_dim1_g,
            self.hidden_dim2_g,
            output_dim=self.x_dim,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        # Set Critic
        self.D = ResDiscriminator(
            self.x_dim,
            self.hidden_dim1_d,
            self.hidden_dim2_d,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        # Optimizers
        self.optim_disc = optim.Adam(self.D.parameters(), lr=self.lr_d, betas=(0.5, 0.9))
        self.optim_gen = optim.Adam(self.G.parameters(), lr=self.lr_g, betas=(0.5, 0.9))

        # Loss function
        self.criterion = nn.BCELoss()


    def generate_fake(self, nb, temperature=1.0):
        """
        Generate fake SNP sequences encoded as 0, 1, 2 using a trained generator.
        Parameters:
            nb (int): the number of samples that you want to generate
            temperature (float): Temperature for Gumbel-Softmax.
        Return:
            converted_AG: a dataframe contains generated SNP sequence encoded by 0,1,2
        """
        self.G.eval()  # Set to evaluation mode
        # Noise
        noise = torch.normal(0,1,size=(nb, self.latent_dim), device=self.device)
        with torch.no_grad():  # Disable gradient calculations
            AG = self.G(noise, temperature=temperature, hard=True)
        # Convert one-hot-like outputs to integer-encoded sequences (0, 1, 2)
        converted_AG = torch.argmax(AG, dim=-1)  
        converted_AG = np.array(converted_AG.cpu())

        return converted_AG

    
    def train(self, 
              TrainDataLoader,
              val_data,
              z_dim: int, 
              epochs: int,
              initial_temp: float = 1.0,
              final_temp: float = 0.1,
              step: int = 10,
              verbose: bool = True,
              checkpoint_dir: str = './checkpoints',
              log_dir: str = './logs'):
        """
        Main train function to train full model.
        ----
        Parameters:
            TrainDataLoader (pytorch loader): train data loader.
            val_data (pd.DataFrame): validation dataframe to compute PCA, precision and recall, etc.
            z_dim (int): latent noise dimension.
            epochs (int): number of training epochs.
            initial_temp (float): initial temperature for Gumbel Softmax function in generator
            final_temp (float): final temperature for Gumbel Softmax function in generator
            step (int): each step to compute loss/metrics.
            verbose (bool): print training callbacks (default True).
            checkpoint_dir (str): where to save model weights.
            log_dir (str): path where to save logs.
        """
        
        # Set up logs and checkpoints
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(log_dir, current_time)

        # Make dir if it does not exist
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            print("Log Directory for current experiment '%s' created" % self.log_dir)

        # Init log path
        self.log_file = os.path.join(self.log_dir, "training_loss_log.txt")
        self.other_metrics_file = os.path.join(self.log_dir, "precision_recall_fst.txt")
        
        with open(self.log_file, "w") as f1:
            # Write header
            f1.write("Epoch\tDiscriminator Loss\tGenerator Loss\n")

        with open(self.other_metrics_file, "w") as f2:
            # Write header
            f2.write("Epoch\tPrecision\tRecall\tFST\n")

        # Define checkpoints path
        self.checkpoint_dir = os.path.join(checkpoint_dir, current_time)
        # Make dir if it does not exist
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
            print("Checkpoint Directory for current experiment '%s' created" % self.checkpoint_dir)

        # Initialize the paths where to store models
        self.G_path = os.path.join(self.checkpoint_dir, "generator")
        if not os.path.exists(self.G_path):
            os.mkdir(self.G_path)
        self.D_path = os.path.join(self.checkpoint_dir, "discriminator")
        if not os.path.exists(self.D_path):
            os.mkdir(self.D_path)

        for epoch in range(epochs):
            self.G.train()
            # Init epoch losses
            total_loss_gen = 0.0
            total_loss_disc = 0.0
            num_batches = len(TrainDataLoader)     
            # Init temperature
            current_temperature = initial_temp - (initial_temp - final_temp) * ((epoch+1) / epochs)
            print("current temperature: "+str(current_temperature))
            
            # Load batches
            for geno in TrainDataLoader:
                # Remove data to the device
                geno = geno.to(self.device)
                batch_size = geno.size(0)

                ################ Train discriminator ################
                self.optim_disc.zero_grad()
                # Labels for real and fake data
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                # Get random latent variables z
                batch_z = torch.normal(0,1,size=(batch_size,z_dim), device=self.device)
                # Generator forward pass 
                gen_outputs = self.G(batch_z, temperature=current_temperature)
                # Forward pass for discriminator on real samples 
                disc_real = self.D(geno)
                real_loss = self.criterion(disc_real, real_labels)
                # Forward pass for discriminator on generated samples
                disc_fake = self.D(gen_outputs.detach())
                fake_loss = self.criterion(disc_fake, fake_labels)

                # Total Discriminator loss
                disc_loss = real_loss + fake_loss
                disc_loss.backward()
                self.optim_disc.step()

                total_loss_disc += disc_loss.detach().item()

                ################ Train generator ################
                self.optim_gen.zero_grad()
                # Generate fake samples and classify them
                batch_z = torch.normal(0,1,size=(batch_size,z_dim), device=self.device)
                gen_outputs = self.G(batch_z, temperature=current_temperature)
                disc_fake = self.D(gen_outputs)
                gen_loss = self.criterion(disc_fake, real_labels)  # Labels as real to trick discriminator
                gen_loss.backward()
                self.optim_gen.step()

                total_loss_gen += gen_loss.detach().item()

            # Print and save all losses for each epoch
            epoch_disc = total_loss_disc/num_batches
            epoch_gen = total_loss_gen/num_batches

            print(f"Epoch [{epoch+1}/{epochs}], Critic Loss: {epoch_disc:.6f}, Generator Loss: {epoch_gen:.6f}")
            # Write loss values to the file after each epoch
            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1}\t{epoch_disc:.6f}\t{epoch_gen:.6f}\n")
                
            # Save the models if the epoch is a checkpoint
            if (epoch+1)%step == 0:
                # Generate synthetic data
                metric_dir = os.path.join(self.log_dir, str(epoch+1))
                os.makedirs(metric_dir, exist_ok=True)
                fake_geno = pd.DataFrame(self.generate_fake(val_data.shape[0], temperature=current_temperature), columns=val_data.columns, dtype=float)
                metrics_result = train_evaluator(val_data, fake_geno, ["precision_recall","fixation_index","pca","allele_freq","geno_freq"], "real", "syn", metric_dir+"/")

                # Write metric
                with open(self.other_metrics_file, "a") as f:
                    f.write(f"{epoch+1}\t{metrics_result['precision']:.6f}\t{metrics_result['recall']:.6f}\t{metrics_result['fixation_index']:.6f}\n")
                
                gen_path = os.path.join(self.G_path, f"g_epoch_{epoch + 1}.pth")
                disc_path = os.path.join(self.D_path, f"d_epoch_{epoch + 1}.pth")
                
                # Save the generator's and discriminator's state dictionaries
                torch.save(self.G.state_dict(), gen_path)
                torch.save(self.D.state_dict(), disc_path)
                print(f"Saved generator and critic at epoch {epoch + 1}")