import os
import sys
import torch

def compare_checkpoints(path1, path2):
    """
    Compara dos archivos de checkpoint de PyTorch/Nerfstudio y verifica
    si sus pesos son idénticos a nivel de bit.
    """
    if not os.path.exists(path1) or not os.path.exists(path2):
        print(f"Error: Uno o ambos archivos no existen.\nPath 1: {path1}\nPath 2: {path2}")
        return False
        
    print(f"Cargando checkpoint 1: {path1}")
    ckpt1 = torch.load(path1, map_location="cpu")
    print(f"Cargando checkpoint 2: {path2}")
    ckpt2 = torch.load(path2, map_location="cpu")
    
    # Nerfstudio guarda los pesos del modelo dentro de la llave 'pipeline'
    state_dict1 = ckpt1.get("pipeline", ckpt1)
    state_dict2 = ckpt2.get("pipeline", ckpt2)
    
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    if keys1 != keys2:
        print("❌ ERROR: Los checkpoints tienen diferentes llaves (arquitecturas distintas).")
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            print(f"Llaves en ckpt1 que no están en ckpt2: {missing_in_2}")
        if missing_in_1:
            print(f"Llaves en ckpt2 que no están en ckpt1: {missing_in_1}")
        return False
        
    mismatches = 0
    total_params = 0
    
    for key in keys1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            continue
            
        total_params += tensor1.numel()
        if not torch.equal(tensor1, tensor2):
            max_diff = (tensor1 - tensor2).abs().max().item()
            print(f"⚠️ DISCREPANCIA en la capa '{key}': diferencia máxima = {max_diff}")
            mismatches += 1
            
    if mismatches == 0:
        print(f"✅ ¡ÉXITO! Los checkpoints son 100% idénticos ({total_params:,} parámetros comparados).")
        return True
    else:
        print(f"❌ FALLÓ: Se encontraron discrepancias en {mismatches} capas.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python3 test_reproducibilidad.py <ruta_ckpt_1> <ruta_ckpt_2>")
    else:
        compare_checkpoints(sys.argv[1], sys.argv[2])
