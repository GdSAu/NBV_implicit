import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils.seed import seed_everything
from models.nerf_basic import BasicNeRF

def test_initialization_determinism():
    seed = 42
    
    seed_everything(seed)
    model1 = BasicNeRF()
    
    seed_everything(seed)
    model2 = BasicNeRF()
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "ERROR: Los pesos inicializados difieren a pesar de usar la misma semilla."
        
    print("✅ Test de inicialización determinista exitoso.")

def test_training_step_determinism():
    seed = 42
    
    # Modelo 1
    seed_everything(seed)
    model1 = BasicNeRF()
    opt1 = optim.Adam(model1.parameters(), lr=1e-3)
    dummy_input1 = torch.rand((10, 6)) # 3 posiciones + 3 vistas
    dummy_target1 = torch.rand((10, 4)) # rgb + alpha (densidad)
    
    output1 = model1(dummy_input1)
    loss1 = nn.MSELoss()(output1, dummy_target1)
    loss1.backward()
    opt1.step()
    
    # Modelo 2
    seed_everything(seed)
    model2 = BasicNeRF()
    opt2 = optim.Adam(model2.parameters(), lr=1e-3)
    dummy_input2 = torch.rand((10, 6)) 
    dummy_target2 = torch.rand((10, 4))
    
    assert torch.allclose(dummy_input1, dummy_input2), "Los inputs aleatorios son distintos."
    assert torch.allclose(dummy_target1, dummy_target2), "Los targets aleatorios son distintos."
    
    output2 = model2(dummy_input2)
    loss2 = nn.MSELoss()(output2, dummy_target2)
    loss2.backward()
    opt2.step()
    
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "ERROR: Los pesos difieren después del primer paso de entrenamiento."
        
    print("✅ Test de entrenamiento determinista exitoso (1 step backward).")

if __name__ == "__main__":
    print("Corriendo tests de determinismo...")
    test_initialization_determinism()
    test_training_step_determinism()
