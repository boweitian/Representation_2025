from typing import List, Union, Optional
from transformers import Pipeline
import torch
import numpy as np
from .rep_readers import DIRECTION_FINDERS, RepReader
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一到 224x224
])

class RepReadingPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if hasattr(outputs, 'encoder_hidden_states') and hasattr(outputs, 'decoder_hidden_states'):
            outputs['hidden_states'] = outputs[f'{which_hidden_states}_hidden_states'] # outputs.keys(): odict_keys(['logits', 'past_key_values', 'hidden_states'])
        # len(outputs['hidden_states']) = 61, outputs['hidden_states'][0].shape = torch.Size([32, 37, 6656])
        hidden_states_layers = {}
        for layer in hidden_layers:
            hidden_states = outputs['hidden_states'][layer]
            hidden_states =  hidden_states[:, rep_token, :].detach()
            if hidden_states.dtype == torch.bfloat16:
                hidden_states = hidden_states.float()
            hidden_states_layers[layer] = hidden_states.detach() # hidden_states_layers[-1]: torch.Size([32, 6656])
        return hidden_states_layers

    def _sanitize_parameters(self, 
                             rep_reader: RepReader=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]


        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"                 
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states
        
        return preprocess_params, forward_params, postprocess_params
 
    def preprocess(self, inputs, processor, image_base_path="data/COCO/val2017/"):
        """
        预处理文本和图片输入，使其适用于 AutoModelForVision2Seq.
        
        Args:
            inputs (dict): 包含 "data"（文本列表）和 "images"（图像文件名列表）
            image_base_path (str): 图像存储的基础路径
            
        Returns:
            dict: 处理后的模型输入，包括 `input_ids`, `attention_mask`, `pixel_values`
        """
        # 加载所有图片
        image = transform(Image.open(image_base_path + inputs["images"]).convert("RGB"))
        # 处理文本 + 图像
        model_inputs = processor(
            text=inputs["data"],     # 文本输入
            images=image,           # 对应的图片输入
            return_tensors="pt",     # 返回 PyTorch 张量
            padding=True,            # 自动填充
            truncation=True          # 适当截断
        )
        return model_inputs

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, pad_token_id=None):
        """
        Args:
        - which_hidden_states (str): Specifies which part of the model (encoder, decoder, or both) to compute the hidden states from. 
                        It's applicable only for encoder-decoder models. Valid values: 'encoder', 'decoder'.
        """
        
        # get model hidden states and optionally transform them with a RepReader
        with torch.no_grad():
            if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"): # pass
                decoder_start_token = [self.tokenizer.pad_token] * model_inputs['input_ids'].size(0)
                decoder_input = self.tokenizer(decoder_start_token, return_tensors="pt").input_ids
                model_inputs['decoder_input_ids'] = decoder_input
            outputs =  self.model(**model_inputs, output_hidden_states=True)
        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)
        # outputs['hidden_states'][0] = torch.Size([32, 37/43, 6656])
        # hidden_states[-1].shape = torch.Size([32, 6656])  len(hidden_states) = 59
        if rep_reader is None:
            return hidden_states
        
        return rep_reader.transform(hidden_states, hidden_layers, component_index) # rep_reader.transform(hidden_states, hidden_layers, component_index)[-1].shape = (32,)


    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, processor, **tokenizer_args):
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(train_inputs, processor = processor, rep_token=rep_token,
            hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        # hidden_states_outputs[0][-1].shape = torch.Size([1, 6656])
        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        return {k: np.vstack(v) for k, v in hidden_states.items()}
    
    def _validate_params(self, n_difference, direction_method):
        # validate params for get_directions
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
            self, 
            train_inputs: Union[str, List[str], List[List[str]]], 
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8, 
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            processor = None,
            **tokenizer_args,):
        """
        Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """
        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
            
        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)
        # {'pca': <class 'repe.rep_readers.PCARepReader'>, 'cluster_mean': <class 'repe.rep_readers.ClusterMeanRepReader'>, 'random': <class 'repe.rep_readers.RandomRepReader'>}
		# if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, processor, **tokenizer_args)
            # len(hidden_states) = 59, hidden_states[-1].shape = (1024, 6656)
            # len(train_inputs) = 1024, train_inputs[0] = "USER: Pretend you're an honest person making statements about the world. ASSISTANT: The"
            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

		# get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)
        return direction_finder
