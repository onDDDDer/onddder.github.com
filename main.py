import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Import your specific model definitions
from torch_nets import tf2torch_inception_v3,tf2torch_inception_v4

# --- Helper Classes ---
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(299),
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        label = self.annotations.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Algorithm 1: Local Flatness Estimation ---
def local_flatness_estimation(model, x_adv, y, num_samples=10, sigma=0.01):
    """
    Calculates flatness based on the std deviation of loss in the neighborhood.
    Returns a tensor of shape [Batch_Size] containing flatness for each image.
    """
    model.eval()
    losses_list = []
    
    # Ensure calculation doesn't track gradients for the estimation itself
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample noise: delta ~ N(0, 0.01^2 I)
            noise = torch.randn_like(x_adv) * sigma
            
            # Clip(x_adv + delta, 0, 1)
            x_perturbed = torch.clamp(x_adv + noise, 0.0, 1.0)
            
            # Compute Loss
            logits = model(x_perturbed)
            # reduction='none' gives us loss per image in the batch
            loss = F.cross_entropy(logits, y, reduction='none') 
            losses_list.append(loss)
    
    # Stack losses: [Batch_Size, num_samples]
    losses_tensor = torch.stack(losses_list, dim=1)
    
    # Calculate Standard Deviation along the sample dimension
    flatness = torch.std(losses_tensor, dim=1)
    
    return flatness

# --- Main Execution ---
if __name__ == "__main__":
    
    # Paths (Configured based on your snippet)
    csv_path = '/data1/dsy/fedpgn/data/images.csv'
    img_root = '/data1/dsy/fedpgn/data/images'
    weight_inc_v3 = '/data1/dsy/fedpgn/torch_nets/tf2torch_inception_v3.npy'
    weight_inc_v4 = '/data1/dsy/fedpgn/torch_nets/tf2torch_inception_v4.npy'
    output_file = 'xxxxxxxv3-v4_results.txt'

    # Dataset Setup
    train_dataset = CustomDataset(csv_file=csv_path, root_dir=img_root)
    BS = 20
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=False, pin_memory=True, num_workers=8)

    # Device Setup
    device = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:9" if torch.cuda.is_available() else "cpu")

    # Model Setup
    # Surrogate Model (Attack Model)
    attackmodel = torch.nn.Sequential(
        Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        tf2torch_inception_v3.KitModel(weight_file=weight_inc_v3).eval().to(device)
    )
    
    # Target Model (Black-box Model)
    targetmodel = torch.nn.Sequential(
        Normalize([0.5,0.5,0.5], [0.5,0.5,0.5]),
        tf2torch_inception_v4.KitModel(weight_file=weight_inc_v4).eval().to(device2)
    )

    # Attack Hyperparameters
    epsilon = 16.0 / 255.0
    T = 10
    alpha = epsilon / T
    mu = 1.0
    delta = 0.5
    zeta = 3.0 * epsilon
    eta1 = epsilon * 0.9
    eta2 = epsilon * 0.3
    
    adv_acc_list = []

    print(f"Starting Attack with T={T}...")

    for batch_idx, (data, target) in enumerate(train_loader):
        # Prepare Data
        x = data.clone().detach().to(device)
        y = target.clone().detach().to(device)
        y_target_device = target.clone().detach().to(device2)
        
        # Initialize Perturbation
        x_adv = x.clone().detach()
        grad = torch.zeros_like(x)
        
        # Storage for Stage 1 and Stage 2 results for Aggregation
        # Will store dictionaries: {'adv': tensor, 'flatness': tensor}
        stage_storage = {} 

        # Multi-stage DMFAA Loop (u=0, u=1, u=2)
        for u in range(3):
            
            # Reset x_adv to the result of the previous stage (or start)
            # But based on your snippet, you restart the loop variable but inherit momentum/state
            # Assuming standard DMFAA: stages are sequential refinements.
            # In your snippet, x_adv resets to x at start of 'u' loop, 
            # but usually, multi-stage attacks refine the previous result.
            # However, respecting your snippet's logic: x_adv = x.clone().detach() inside u loop
            # meant you were generating 3 independent variants? 
            # EDIT: Based on context, we need the result at the end of the stage.
            # I will keep the loop structure but ensure we capture the final x_adv of each stage.
            x_adv = x.clone().detach() 

            # Set sampling count based on stage
            if u == 0:
                N_samples = 20
            else:
                N_samples = 10
            
            # Inner Optimization Loop
            for t in range(T):
                g_bar = torch.zeros_like(x)
                
                # --- Diversified Sampling (ODI/OODI) ---
                for i in range(N_samples):
                    x_prime = x_adv.detach()
                    batch_size = x_adv.size(0)
                    num_classes = 1001

                    # Random direction w
                    w = torch.empty((batch_size, num_classes), device=device).uniform_(-1, 1)
                    
                    x_prime = x_prime.detach().requires_grad_(True)
                    
                    # 1. Output Diversity Gradient (g_odi)
                    f_x_prime = attackmodel(x_prime)
                    out_logits = (f_x_prime * w).sum(dim=1).mean()
                    
                    if x_prime.grad is not None:
                        x_prime.grad.zero_()
                    attackmodel.zero_grad()
                    out_logits.backward(retain_graph=True)
                    g_k = x_prime.grad.detach().clone()
                    
                    # 2. Classification Gradient (h_cls)
                    x_prime.grad.zero_()
                    logits = attackmodel(x_prime)
                    loss_cls = F.cross_entropy(logits, y)
                    loss_cls.backward()
                    h_k = x_prime.grad.detach().clone()
                    
                    # 3. Orthogonal Projection
                    g_k_flat = g_k.view(batch_size, -1)
                    h_k_flat = h_k.view(batch_size, -1)
                    
                    proj = (h_k_flat * g_k_flat).sum(dim=1, keepdim=True) / \
                           (g_k_flat.norm(p=2, dim=1, keepdim=True).pow(2) + 1e-12)
                    
                    g_perp_flat = h_k_flat - proj * g_k_flat
                    g_perp = g_perp_flat.view_as(x_prime)
                    
                    g_odi = g_k 
                    g_ortho = g_perp
                    # 4. Update x_prime
                    update = eta1 * g_odi.sign() + eta2 * g_ortho.sign()
                    x_prime = x_prime + update
                    
                    # 5. Project back to L_inf ball around x_adv
                    control = torch.clamp(x_prime - x_adv, min=-epsilon, max=epsilon)
                    x_prime = torch.clamp(x_adv + control, min=0.0, max=1.0).detach()
                    
                    # 6. Add Random Noise (Transformation)
                    x_prime = x_prime + torch.FloatTensor(np.random.uniform(-zeta, zeta, size=x.shape)).to(device)
                    x_prime = x_prime.detach().requires_grad_(True)

                    # --- Gradient Calculation for Momentum ---
                    outputA = attackmodel(x_prime)
                    cost = F.cross_entropy(outputA, y)
                    cost.backward()
                    grad_prime = x_prime.grad
                    
                    # Nesterov lookahead
                    x_star = (x_prime - alpha * grad_prime / (torch.norm(grad_prime, p=1) + 1e-8)).detach().requires_grad_(True)
                    output2A = attackmodel(x_star)
                    costt = F.cross_entropy(output2A, y)
                    costt.backward()
                    grad_star = x_star.grad
                    
                    # Accumulate Gradient
                    g_bar += (1 / N_samples) * ((1 - delta) * grad_prime.float() + delta * grad_star.float())

                # Update Momentum and Image
                grad = mu * grad + g_bar / (torch.norm(g_bar, p=1) + 1e-8)
                x_adv = x_adv.detach() + alpha * grad.sign()
                
                # Projection
                control = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x + control, min=0, max=1).detach()
            
            # --- End of Stage 'u' ---
            # If stage is 1 or 2, calculate flatness and store results
            if u in [1, 2]:
                # Algorithm 1: Local Flatness Estimation
                # We use the final x_adv of this stage
                flat_score = local_flatness_estimation(attackmodel, x_adv, y)
                stage_storage[u] = {
                    'adv': x_adv.detach(), 
                    'flatness': flat_score
                }

        # --- Flatness-Aware Aggregation (FAA) ---
        # Equation: lambda_s = exp(-flatness_s) / sum(exp(-flatness_k))
        # We aggregate results from stage 1 and stage 2
        
        flatness1 = stage_storage[1]['flatness']
        flatness2 = stage_storage[2]['flatness']
        
        # Calculate weights (Softmax on negative flatness)
        # Note: Calculation is per-sample in the batch
        exp_neg_f1 = torch.exp(-flatness1)
        exp_neg_f2 = torch.exp(-flatness2)
        sum_exp = exp_neg_f1 + exp_neg_f2
        
        lambda1 = exp_neg_f1 / sum_exp
        lambda2 = exp_neg_f2 / sum_exp
        
        # Reshape lambdas for broadcasting: [Batch] -> [Batch, 1, 1, 1]
        lambda1 = lambda1.view(-1, 1, 1, 1)
        lambda2 = lambda2.view(-1, 1, 1, 1)
        
        # Weighted Sum
        x_final = lambda1 * stage_storage[1]['adv'] + lambda2 * stage_storage[2]['adv']
        
        # Ensure final constraint
        control = torch.clamp(x_final - x, min=-epsilon, max=epsilon)
        x_final = torch.clamp(x + control, min=0, max=1).detach()

        # --- Evaluation on Target Model ---
        adversarial_samples = x_final.to(device2)
        outputs_target = targetmodel(adversarial_samples)
        _, predicted_labels = torch.max(outputs_target.data, 1)
        
        # Calculate Batch Accuracy
        adv_acc = (predicted_labels != y_target_device).sum().item() / target.size(0)
        adv_acc_list.append(adv_acc)
        
        # --- Save Batch Result ---
        with open(output_file, 'a') as f:
            f.write(f"Batch {batch_idx}: Attack Rate {adv_acc:.4f}\n")

    # --- Final Statistics ---
    adv_acc_list = np.array(adv_acc_list)
    mean_acc = np.mean(adv_acc_list)
    
    print(f"Final Attack Success Rate: {mean_acc:.4f}")
    
    with open(output_file, 'a') as f:
        f.write(f"Final Average Attack Rate: {mean_acc:.4f}\n")