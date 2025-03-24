

import torch 


def load_model(model_config,model_class ,device, saved_model_path):
    model = model_class(model_config,2)
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model = model.to(device)
    return model