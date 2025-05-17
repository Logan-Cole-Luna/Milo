import torch
import torch.nn as nn

# Define a residual block with layer normalization
class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, in_features),
            nn.LayerNorm(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

# Define a more complex policy network
class ComplexPolicyNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ComplexPolicyNetwork, self).__init__()
        
        # Model size parameters
        input_dim = observation_space.shape[0]
        hidden_dim = 256  # Increased size
        output_dim = action_space.shape[0]
        n_residual_blocks = 4  # More residual blocks
        
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Residual blocks for increased depth
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim // 2) 
            for _ in range(n_residual_blocks)
        ])
        
        # Output layers with bottleneck
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
        # Better initialization
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization for better convergence
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Make the last layer produce smaller outputs
        last_layer = self.output_layers[-1]
        if isinstance(last_layer, nn.Linear):
            last_layer.weight.data.mul_(0.1)

    def forward(self, x):
        x = self.input_embedding(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
            
        # Output processing
        x = self.output_layers(x)
        
        # Scale to appropriate action range
        x = torch.tanh(x) * 0.1  # Small actions for more controlled movement
        return x
