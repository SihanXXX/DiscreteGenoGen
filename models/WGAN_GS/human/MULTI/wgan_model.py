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

# Residual Block for Generator and Critic
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

class ResNetBlockCritic(nn.Module):
    def __init__(self, input_dim, alpha=0.05):
        super(ResNetBlockCritic, self).__init__()
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
                 sex_dim: int,
                 height_dim: int, 
                 hidden_dim1: int,
                 hidden_dim2: int,
                 hidden_dim3: int,
                 output_dim: int,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()

        """Parameters:
            latent_dim (int): dimension of latent noise vector z
            sex_dim (int): sex label (1 for men, 0 for women)
            height_dim (int): numerical variable for phenotype
            hidden_dim1 (int): dimension of 1st hidden layer
            hidden_dim2 (int): dimension of 2nd hidden layer
            hidden_dim3 (int): dimension of 3rd hidden layer
            output_dim (int): dimension of generated data (nb_SNPs)
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """
        # Dimensions
        self.latent_dim = latent_dim
        self.sex_dim = sex_dim
        self.height_dim = height_dim
        self.input_dim = self.latent_dim + 2 * self.sex_dim + 2 * self.height_dim
        # Layers params
        self.output_dim = output_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dims=[hidden_dim1, hidden_dim2, hidden_dim3]

        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Initial Layer (Before ResNet Block)
        self.initial = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),  
            self.activation_func
        )

        # (n_hidden_dim - 1) ResNet block
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.res_blocks.append(nn.Sequential(
                ResNetBlockGenerator(self.hidden_dims[i] + 2 * self.sex_dim + 2 * self.height_dim, self.negative_slope),
                nn.Linear(self.hidden_dims[i]+2*self.sex_dim+2*self.height_dim, self.hidden_dims[i+1]),  # Transition layer
                nn.LeakyReLU(self.negative_slope) 
            ))

        #final layer
        self.final = nn.Linear(self.hidden_dims[-1]+2*self.sex_dim+2*self.height_dim, 3*self.output_dim)
            

    def forward(self, z: torch.tensor, sex: torch.tensor, height: torch.tensor, temperature = 1.0, hard = False):
        """ Main function to generate from input noise vector.
        ----
        Parameters:
            x (torch.tensor): input noise vector
            sex (torch.tensor): torch tensor of dimension 1
            height (torch.tensor): torch tensor of dimension 1
            temperature (float): The temperature parameter (τ) in Gumbel-Softmax controls the sharpness of the output distribution
            hard (bool): The hard parameter in Gumbel-Softmax determines whether the output is a soft probability distribution (hard=False during training)
            or a one-hot discrete vector (hard=True during inference)
        Returns:
            (torch.tensor): generated data
        """
        x = torch.cat((height, sex, z, sex, height), 1)
        x = self.initial(x)
        for block in self.res_blocks:
            x = torch.cat((height, sex, x, sex, height), 1)
            x = block(x)
        x = torch.cat((height, sex, x, sex, height), 1)
        x = self.final(x)
        logits = x.view(-1, self.output_dim, 3)  # Reshape to [BATCH_SIZE, seq_length, 3]
        output = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        return output

class ResCritic(nn.Module):
    """ Critic """
    def __init__(self, x_dim: int,
                 sex_dim: int,
                 height_dim: int,
                 hidden_dim1: int,
                 hidden_dim2: int,
                 hidden_dim3: int,
                 activation_func: str = 'relu',
                 negative_slope: float = 0.1):
        super().__init__()
        """Parameters:
            x_dim (int): dimension of SNP sequence length
            sex_dim (int): dimension of sex label.
            height_dim (int): dimension of height variable
            hidden_dim1 (int): dimension of 1st hidden layer.
            hidden_dim2 (int): dimension of 2nd hidden layer.
            hidden_dim3 (int): dimension of 3rd hidden layer.
            activation_func (str): activation function between hidden layers (either 'relu' or 'leaky_relu'). Default 'relu'.
            negative_slope (float): slope of leaky relu activation. Default 0.1.
        """
        # Layers params
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dims=[hidden_dim1, hidden_dim2, hidden_dim3]
        
        # Dimensions
        self.x_dim = x_dim
        self.sex_dim = sex_dim
        self.height_dim = height_dim
        
        # Activation function
        self.negative_slope = negative_slope
        if activation_func.lower() == 'leaky_relu':
            self.activation_func = nn.LeakyReLU(self.negative_slope)
        elif activation_func.lower() == 'relu':
            self.activation_func = nn.ReLU()

        # Initial Layer
        self.initial = nn.Sequential(nn.Linear(self.x_dim *3+2*self.sex_dim+2*self.height_dim, self.hidden_dims[0]), nn.LeakyReLU(self.negative_slope))

        # ResBlock Layer
        self.res_blocks = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            self.res_blocks.append(nn.Sequential(
                ResNetBlockCritic(self.hidden_dims[i]+2*self.sex_dim+2*self.height_dim, self.negative_slope),
                nn.Linear(self.hidden_dims[i]+2*self.sex_dim+2*self.height_dim, self.hidden_dims[i+1]),  # Transition layer
                nn.LeakyReLU(self.negative_slope) 
            ))

        # Final Layer
        self.final = nn.Linear(self.hidden_dims[-1], 1)  # Scalar output for Wasserstein distance

    def forward(self, x, sex: torch.tensor, height: torch.tensor):
        x = x.view(x.size(0), -1) # Flatten Input
        x = torch.cat((height, sex, x, sex, height), 1)
        x = self.initial(x)
        for block in self.res_blocks:
            x = torch.cat((height, sex, x, sex, height), 1)
            x = block(x)
        return self.final(x)

class WGAN_GP(object):
    """
    Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) class.
    """
    def __init__(self, config: dict, device: torch.device):
        """ Parameters:
            config (dict): model architecture dictionary
            device: which CPU or GPU(0,1,2) to use
        """        
        # Set Architecture
        self.latent_dim = config['latent_dim']
        self.sex_dim = config['sex_dim']
        self.height_dim = config['height_dim']
        self.x_dim = config['x_dim']
        self.hidden_dim1_g = config['hidden_dim1_g']
        self.hidden_dim2_g = config['hidden_dim2_g']
        self.hidden_dim3_g = config['hidden_dim3_g']
        self.hidden_dim1_d = config['hidden_dim1_d']
        self.hidden_dim2_d = config['hidden_dim2_d']
        self.hidden_dim3_d = config['hidden_dim3_d']
        self.lr_g = config['lr_g']
        self.lr_d = config['lr_d']

        # Set device
        self.device = device

        # Set Generator
        self.G = ResGenerator(
            self.latent_dim,
            self.sex_dim,
            self.height_dim,
            self.hidden_dim1_g,
            self.hidden_dim2_g,
            self.hidden_dim3_g,
            output_dim=self.x_dim,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        # Set Critic
        self.D = ResCritic(
            self.x_dim,
            self.sex_dim,
            self.height_dim,
            self.hidden_dim1_d,
            self.hidden_dim2_d,
            self.hidden_dim3_d,
            activation_func=config['activation'],
            negative_slope=config['negative_slope']).to(
            self.device)

        # Optimizers
        self.optim_disc = optim.Adam(self.D.parameters(), lr=self.lr_d, betas=(0.5, 0.9))
        self.optim_gen = optim.Adam(self.G.parameters(), lr=self.lr_g, betas=(0.5, 0.9))

        # Learning rate schedulers
        # self.gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim_gen, mode='max', factor=0.5, patience=10, verbose=True)
        # self.disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optim_disc, mode='max', factor=0.5, patience=10, verbose=True)

        # Loss function
        self.lambda_gp = config['lambda_penalty']
        self.n_critic = config['n_critic']

    def gradient_penalty(
            self,
            real_data: torch.tensor,
            fake_data: torch.tensor,
            sex: torch.tensor,
            height: torch.tensor):
        """
        Compute gradient penalty.
        ----
        Parameters:
            real_data (torch.tensor): real data
            fake_data (torch.tensor): generated data
            sex (torch.tensor): sex condition
            height (torch.tensor): height condition
        Returns:
            gp (torch.tensor): gradient penalty i.e mean squared gradient norm on interpolations (||Grad[D(x_inter)]||2 - 1)^2
        """
        # Fixed batch size
        BATCH_SIZE = real_data.size(0)

        # Random weight term for interpolation between real and fake sample
        alpha = torch.rand(BATCH_SIZE, 1, 1, requires_grad=True, device=real_data.device)

        # Interpolation between real data and fake data.
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

        # Get outputs from critic
        d_interpolates = self.D(interpolates, sex, height)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs = d_interpolates,
            inputs = interpolates,
            grad_outputs = torch.ones_like(d_interpolates, device=real_data.device),
            create_graph = True,
            retain_graph = True)[0]

        # Compute the gradient penalty
        gradients = gradients.view(BATCH_SIZE, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def get_crit_loss(self, crit_fake_pred, crit_real_pred, gp):
        '''
        Return the loss of a critic given the critic's scores for fake and real SNP sequences, the gradient penalty, and gradient penalty weight.
        Parameters:
            crit_fake_pred: the critic's scores of the fake SNP sequences
            crit_real_pred: the critic's scores of the real SNP sequences
            gp: the unweighted gradient penalty
        Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
        '''
        crit_loss = torch.mean(crit_fake_pred - crit_real_pred) + self.lambda_gp * gp
        return crit_loss

    def get_gen_loss(self, crit_fake_pred):
        '''
        Return the loss of a generator given the critic's scores of the generator's fake SNP sequences.
        Parameters:
            crit_fake_pred: the critic's scores of the fake SNP sequences
        Returns:
            gen_loss: a scalar loss value for the current batch of the generator
    '''
        gen_loss = -torch.mean(crit_fake_pred)
        return gen_loss

    def generate_fake(self, nb, sex, height, temperature=1.0, batch_size=2000):
        """
        Generate fake SNP sequences encoded as 0, 1, 2 using a trained generator.
        Parameters:
            nb (int): the number of samples that you want to generate
            sex (torch.tensor): sex condition
            height (torch.tensor): height condition
            temperature (float): Temperature for Gumbel-Softmax.
            batch_size (int): The size of the batches to generate to prevent memory issues.
        Return:
            converted_AG: a dataframe contains generated SNP sequence encoded by 0,1,2
        """
        trained_model = self.G
        trained_model.eval()  # Set to evaluation mode
        all_generated_samples = []  # Initialize an empty list to store the generated sequences

        # Loop through mini-batches to generate the required number of samples
        for start_idx in range(0, nb, batch_size):
            end_idx = min(start_idx + batch_size, nb)  # Ensure we don't go beyond nb
            current_batch_size = end_idx - start_idx
            # Generate noise for the current batch
            noise = torch.normal(0,1,size=(current_batch_size, self.latent_dim), device=self.device)
            # Select the corresponding phenotype conditions for this batch
            current_sex = sex[start_idx:end_idx]
            current_height = height[start_idx:end_idx]
            # Disable gradient calculations for memory efficiency
            with torch.no_grad():
                AG = trained_model(noise, current_sex, current_height, temperature=temperature, hard=True)
            # Convert one-hot-like outputs to integer-encoded sequences (0, 1, 2)
            converted_batch = torch.argmax(AG, dim=-1).cpu().numpy()  # Move to CPU and convert to numpy
            all_generated_samples.append(converted_batch)

        # Concatenate all batches into one large array
        converted_AG = np.concatenate(all_generated_samples, axis=0)

        return converted_AG

    def train(self, 
              TrainDataLoader,
              val_data_geno_df,
              val_data_sex_tensor,
              val_data_height_tensor,
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
            val_data_geno_df (pd.DataFrame): validation geno dataframe.
            val_data_sex_tensor (torch.tensor): validation sex torch tensor. 
            val_data_height_tensor (torch.tensor): validation height torch tensor. 
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
            f1.write("Epoch\tDiscriminator Loss\tGenerator Loss\tGradient Penalty\n")

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

        best_f1 = 0.0
        for epoch in range(epochs):
            self.G.train()
            # Init epoch losses
            total_loss_gen = 0.0
            total_loss_critic = 0.0
            total_gp = 0.0
            num_batches = len(TrainDataLoader)     
            # Init temperature
            current_temperature = initial_temp - (initial_temp - final_temp) * ((epoch+1) / epochs)
            print("current temperature:")
            print(current_temperature)
            
            # Load batches
            for geno, sex, height in TrainDataLoader:
                # Batch losses
                batch_loss_critic = 0.0
                batch_gp = 0.0
                
                # Remove data to the device
                geno = geno.to(self.device)
                sex = sex.to(self.device)
                height = height.to(self.device)
                batch_size = geno.size(0)

                ################ Train discriminator ################
                for _ in range(self.n_critic) : 
                    for param in self.D.parameters():
                        param.requires_grad = True

                    # Get random latent variables z
                    batch_z = torch.normal(0,1,size=(batch_size,z_dim), device=self.device)
                    # Generator forward pass 
                    gen_outputs = self.G(batch_z, sex, height, temperature=current_temperature)

                    # Forward pass for discriminator on real samples 
                    crit_real = self.D(geno, sex, height)
                
                    # Forward pass for discriminator on generated samples
                    crit_fake = self.D(gen_outputs.detach(), sex, height)
                
                    # Compute gradient penalty term
                    gp = self.gradient_penalty(geno, gen_outputs, sex, height)

                    # Critic loss
                    crit_loss = self.get_crit_loss(crit_fake, crit_real, gp)
                    # Compute batch statistics
                    batch_loss_critic += crit_loss.detach().item()
                    batch_gp += gp.detach().item()

                    # Update Critic
                    self.optim_disc.zero_grad(set_to_none=True) # Zero out the gradient before backpropagation
                    crit_loss.backward(retain_graph=True) # Update gradients
                    self.optim_disc.step()

                total_loss_critic += (batch_loss_critic/self.n_critic)
                total_gp += (batch_gp/self.n_critic)

                ################ Train generator ################
                for param in self.D.parameters():
                    param.requires_grad = False
                # Get random latent variables z
                batch_z = torch.normal(0,1,size=(batch_size,z_dim), device=self.device)
                # Generator forward pass
                gen_outputs = self.G(batch_z, sex, height, temperature=current_temperature)
                crit_fake = self.D(gen_outputs, sex, height)
                # Generator loss
                gen_loss = self.get_gen_loss(crit_fake)

                self.optim_gen.zero_grad()
                gen_loss.backward() # Update gradients
                self.optim_gen.step() # Update optimizer

                total_loss_gen += gen_loss.detach().item()

            # Pring and save all losses for each epoch
            epoch_crit = total_loss_critic/num_batches
            epoch_gen = total_loss_gen/num_batches
            epoch_gp = total_gp/num_batches
            print(f"Epoch [{epoch+1}/{epochs}], Critic Loss: {epoch_crit:.4f}, Generator Loss: {epoch_gen:.4f}, Gradient Penalty: {epoch_gp:.4f}")
            # Write loss values to the file after each epoch
            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1}\t{epoch_crit:.4f}\t{epoch_gen:.4f}\t{epoch_gp:.4f}\n")

            # checkpoint
            if (epoch+1)%step == 0:
                # Generate synthetic data
                metric_dir = os.path.join(self.log_dir, str(epoch+1))
                os.makedirs(metric_dir, exist_ok=True)
                fake_geno = pd.DataFrame(self.generate_fake(val_data_geno_df.shape[0], val_data_sex_tensor, val_data_height_tensor, temperature=current_temperature), columns=val_data_geno_df.columns, dtype=float)
                metrics_result = train_evaluator(val_data_geno_df, fake_geno, ["precision_recall","fixation_index","pca","geno_freq"], "real", "syn", metric_dir+"/")

                if (metrics_result['precision'] + metrics_result['recall']) != 0:
                    f1_score=2*(metrics_result['precision']*metrics_result['recall'])/(metrics_result['precision']+metrics_result['recall'])
                else:
                    f1_score = 0

                # Write metric
                with open(self.other_metrics_file, "a") as f:
                    f.write(f"{epoch+1}\t{metrics_result['precision']:.4f}\t{metrics_result['recall']:.4f}\t{metrics_result['fixation_index']:.4f}\n")

                if f1_score >= best_f1:  # Save the best model if the F1 score improved
                    best_f1 = f1_score
                    best_G_path = os.path.join(self.G_path, "best_G.pth")
                    torch.save(self.G.state_dict(), best_G_path)
                    print(f"New best model saved at epoch {epoch+1} with F1 score {f1_score:.6f}")
                else:
                    gen_path = os.path.join(self.G_path, f"g_epoch_{epoch + 1}.pth")
                    disc_path = os.path.join(self.D_path, f"d_epoch_{epoch + 1}.pth")
                    # Save the generator's and discriminator's state dictionaries
                    torch.save(self.G.state_dict(), gen_path)
                    torch.save(self.D.state_dict(), disc_path)
                    print(f"Saved generator and critic at epoch {epoch + 1}")