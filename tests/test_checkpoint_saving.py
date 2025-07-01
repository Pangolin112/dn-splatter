import torch

# Replace with the actual path to your checkpoint
checkpoint_path = "outputs/unnamed/dn-splatter/2025-06-10_190528/nerfstudio_models/step-000001500.ckpt"
checkpoint = torch.load(checkpoint_path)

# Check the top-level keys
print("Checkpoint keys:", checkpoint.keys())

# Inspect the optimizer state
if "optimizers" in checkpoint:
    print("\nOptimizer states:")
    for opt_name, opt_state in checkpoint["optimizers"].items():
        print(f"- {opt_name}:")
        print("  Keys:", opt_state.keys())
        if "state" in opt_state:
            print("  Parameter states:")
            for param_id, param_state in opt_state["state"].items():
                print(f"    Parameter {param_id}: {param_state.keys()}")