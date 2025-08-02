import torch
import matplotlib.pyplot as plt

def get_model_summary(model):
    summary = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            weight = getattr(module, "weight", None)
            if weight is not None:
                sparsity = 100. * float((weight == 0).sum()) / weight.numel()
                summary[name] = {
                    "shape": tuple(weight.shape),
                    "sparsity": round(sparsity, 2)
                }
    return summary

def diff_models(original_summary, pruned_summary):
    diff = []
    for layer in original_summary:
        if layer in pruned_summary:
            orig = original_summary[layer]
            prun = pruned_summary[layer]
            if orig["shape"] != prun["shape"] or orig["sparsity"] != prun["sparsity"]:
                diff.append({
                    "layer": layer,
                    "orig_shape": orig["shape"],
                    "pruned_shape": prun["shape"],
                    "orig_sparsity": orig["sparsity"],
                    "pruned_sparsity": prun["sparsity"]
                })
    return diff

def plot_layer_sparsity(diff):
    layers = [d['layer'] for d in diff]
    sparsity = [d['pruned_sparsity'] for d in diff]

    plt.figure(figsize=(12, 6))
    plt.barh(layers, sparsity)
    plt.xlabel("Pruned Sparsity (%)")
    plt.title("Layer-wise Sparsity After Structured Pruning")
    plt.tight_layout()
    plt.show()