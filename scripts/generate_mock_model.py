import torch
from typing import Dict, List, Tuple

class MockPotential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simple harmonic potential constants
        self.k = 10.0 # eV/A^2
        self.r0 = 2.0 # Angstrom

    def forward(self, z: torch.Tensor, pos: torch.Tensor, cell: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Simple dummy potential: harmonic spring between first two atoms
        # Ignore Z and Cell for this test
        
        # Calculate distance between atom 0 and 1
        r_vec = pos[0] - pos[1]
        r = torch.norm(r_vec)
        
        # Energy = 0.5 * k * (r - r0)^2
        energy = 0.5 * self.k * (r - self.r0)**2
        
        # Force magnitude = -k * (r - r0) (attractive if r > r0)
        # Vector force on 0 = F_mag * (r_vec / r)
        # Force on 1 = -Force on 0
        
        # We need gradients. Let's use autograd to be sure.
        # But for TorchScript export without tracing, explicit calculation is better if possible.
        # However, for MLIPs usually they use autograd. 
        # But here we just want to return values.
        
        f_mag = -self.k * (r - self.r0)
        n = r_vec / r
        f0 = f_mag * n
        f1 = -f0
        
        forces = torch.zeros_like(pos)
        forces[0] = f0
        forces[1] = f1
        
        # Virial (approx for two body)
        # W_alpha_beta = sum_i r_i_alpha * f_i_beta
        # This definition varies. Lumol uses:
        # W = sum_i f_i (outer) r_i  <-- Check this definition in Lumol source!
        # lumol::Virial definition:
        # let fact = self.force(r.norm());
        # let rn = r.normalized();
        # let force = fact * rn;
        # force.tensorial(r)  <-- This is f (outer) r
        
        # Let's compute f (outer) r for the pair
        # W = f0 (outer) pos[0] + f1 (outer) pos[1]
        # Since f1 = -f0
        # W = f0 (outer) (pos[0] - pos[1]) = f0 (outer) r_vec
        
        virial = torch.outer(f0, r_vec)
        
        return {
            "energy": energy,
            "forces": forces,
            "virial": virial
        }

if __name__ == "__main__":
    model = MockPotential()
    scripted_model = torch.jit.script(model)
    scripted_model.save("mock_mlip.pt")
    print("Saved mock_mlip.pt")
