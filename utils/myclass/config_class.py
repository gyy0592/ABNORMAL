from ..myfuncs.histSeries import beta_extract_func,beta_wavelet

class FileConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if key == 'extract_func':
                self.extract_func = globals().get(value)
                if not callable(self.extract_func):
                    raise ValueError(f"Function '{value}' not found in global scope")
            elif key == 'dataset':
                self.dataset = globals().get(value)
                if not callable(self.dataset):
                    raise ValueError(f"Dataset function '{value}' not found in global scope")
            else:
                setattr(self, key, value)


class Config:
    def __init__(self, config_dict):
        self._configs = []
        for key, value in config_dict.items():
            if isinstance(value, dict):
                nested_config = Config(value)
                self._configs.append(nested_config)
            else:
                for key, value in config_dict.items():
                    if key == 'extract_func':
                        self.extract_func = globals().get(value)
                        if not callable(self.extract_func):
                            raise ValueError(f"Function '{value}' not found in global scope")
                    elif key == 'dataset':
                        self.dataset = globals().get(value)
                        if not callable(self.dataset):
                            raise ValueError(f"Dataset function '{value}' not found in global scope")
                    else:
                        setattr(self, key, value)
    def __getitem__(self, key):
        if isinstance(key, int):
            return self._configs[key]
        else:
            return getattr(self, key)

    def __iter__(self):
        return iter(self._configs)

