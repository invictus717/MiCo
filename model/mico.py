"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
from builtins import NotImplementedError
import copy
from .transformer import GELU
import torch
import math
from torch import nn 
from torch.nn import LayerNorm as FusedLayerNorm
from torch.nn import LayerNorm
import torch.nn.functional as F
from torchvision.transforms import *
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    
class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)
    def forward(self, cls_token):
        return self.linear(cls_token)


class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)
    def forward(self, cls_token):
        return self.linear2(self.layernorm(self.activation(self.linear1(cls_token))))


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
   

class MMGeneralModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def construct_vision_encoder(self):
        ##### construct vision encoder 
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            self.load_clip_model() 
        elif self.config.vision_encoder_type.startswith('swin'):
            self.load_swin_model()        
        elif self.config.vision_encoder_type.startswith('videoswin'):
            self.load_videoswin_model()          
        else:
            raise NotImplementedError
            
    def construct_audio_encoder(self):
        self.audio_dim = self.vision_dim

    def construct_depth_encoder(self):
        self.depth_dim = self.vision_dim
   
    def construct_multimodal_encoder(self):    
        from model.bert import BertForMaskedLM
        from transformers.models.bert.configuration_bert import BertConfig
        from transformers import BertTokenizer
        bertconfig = BertConfig.from_pretrained("./model/bert-base-uncased-crossattn")
        self.multimodal_encoder = BertForMaskedLM(bertconfig)
        self.multimodal_dim = 768

        if self.config.checkpointing:
            self.multimodal_encoder._set_gradient_checkpointing(self.multimodal_encoder.bert.encoder, True)

        self.multimodal_encoder.tokenizer = BertTokenizer.from_pretrained('./model/tokenizer')
        self.multimodal_encoder.tokenizer.bos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.multimodal_encoder.tokenizer.eos_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.multimodal_encoder.tokenizer.pad_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        self.multimodal_encoder.tokenizer.mask_token_id = self.multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
    
    def forward_vision_encoder(self, vision_pixels):   ### b,n,3,h,w
        
        b,n,_,h,w = vision_pixels.shape
        
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):

            vision_output = self.vision_encoder.visual(vision_pixels.reshape(b*n,3,h,w), return_all_features=True)
            vision_output = vision_output.reshape(b,-1,*vision_output.shape[-2:])
  
        
        elif self.config.vision_encoder_type.startswith('swin'):
            vision_output = self.vision_encoder(vision_pixels.reshape(b*n,3,h,w))
            vision_output = vision_output.reshape(b,-1,*vision_output.shape[-2:])
    
        elif self.config.video_encoder_type.startswith('videoswin'):
            vision_output = self.vision_encoder(vision_pixels.transpose(1,2)) ### b,c,n,h,w
            vision_output = vision_output.permute(0, 2, 3, 4, 1)  ###b,n,h,w,c
            vision_output = vision_output.reshape(b,n,-1,vision_output.shape[-1])

        else:
            raise NotImplementedError()

        return vision_output  #### B , n , x ,C  n = self.frame_num

    def forward_audio_encoder(self, audio_spectrograms):      
        audio_spectrograms = audio_spectrograms.unsqueeze(2).repeat(1,1,3,1,1)
        audio_output = self.forward_vision_encoder(audio_spectrograms)
            
        return audio_output

    def forward_depth_encoder(self, depth_pixels):     
        depth_output = self.forward_vision_encoder(depth_pixels)

        return depth_output

    def forward_multimodal_encoder(self, input_ids, attention_mask, condition_feat=None, labels=None, position_ids=None, preprocess=True):
        return self.multimodal_encoder(input_ids = input_ids,
                                        attention_mask = attention_mask,
                                        encoder_hidden_states=condition_feat,
                                        labels = labels
                                        )

    def pool_vision_for_contra(self, feature):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            feature = feature[:,:,0]
        elif self.config.vision_encoder_type.startswith('swin'):
            feature = feature.mean(dim=2)
        feature = torch.mean(feature, dim=1)
        return feature

    def pool_audio_for_contra(self, feature):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            feature = feature[:,:,0]
        elif self.config.vision_encoder_type.startswith('swin'):
            feature = feature.mean(dim=2)
        feature = torch.mean(feature, dim=1)
        return feature

    def pool_depth_for_contra(self, feature):  #feature b ,n ,x ,c
        #### always use frame_avg  for retrieval
        if self.config.vision_encoder_type.startswith('clip') or self.config.vision_encoder_type.startswith('evaclip'):
            feature = feature[:,:,0]
        elif self.config.vision_encoder_type.startswith('swin'):
            feature = feature.mean(dim=2)
        feature = torch.mean(feature, dim=1)
        return feature

    def pool_text_for_contra(self, feature):  #feature b ,n ,x, c
        return feature[:,0]

    def get_multimodal_forward_input_vision(self, vision_output):
        b,n,x,c = vision_output.shape

        if self.config.pool_video:
            vision_output = torch.cat([vision_output[:,:,0:1], vision_output[:,:,1:].mean(2, keepdim=True)], dim=2)
            
        vision_output = self.hidden_trans_vision_multimodal(vision_output)  

        if self.config.frame_embedding_type == 'adaptive':
            if n!=self.vision_frame_embedding.shape[1]: #### testing and interpolate
                # dtype = self.vision_frame_embedding.dtype
                vision_frame_embedding = F.interpolate(self.vision_frame_embedding.float().permute(0,2,1),n,mode='nearest').permute(0,2,1).to(self.vision_frame_embedding)
            else:
                vision_frame_embedding = self.vision_frame_embedding
            vision_output =  vision_output + vision_frame_embedding.unsqueeze(-2)

        elif self.config.frame_embedding_type == 'none':
            pass

        vision_output =  vision_output.reshape(b,-1,self.multimodal_dim)

        if hasattr(self,'vision_type_embeddings'): ### for three modality
            vision_output =  vision_output + self.vision_type_embeddings

        # vision_output = vision_output[:,:450]
        return vision_output
    
    def get_multimodal_forward_input_audio(self, audio_output):
        b,n,x,c = audio_output.shape

        if self.config.pool_video:
            audio_output = torch.cat([audio_output[:,:,0:1], audio_output[:,:,1:].mean(2, keepdim=True)], dim=2)
      
        if n!= self.audio_frame_embedding.shape[1]: #### testing and interpolate
            audio_frame_embedding = F.interpolate(self.audio_frame_embedding.permute(0,2,1),n,mode='nearest').permute(0,2,1)
        else:
            audio_frame_embedding = self.audio_frame_embedding
        audio_output = self.hidden_trans_audio_multimodal(audio_output)
        audio_output =  audio_output + audio_frame_embedding.unsqueeze(-2)
        audio_output = audio_output.reshape(b,-1,self.multimodal_dim)
        audio_output = audio_output + self.audio_type_embeddings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        return audio_output

    def get_multimodal_forward_input_depth(self, depth_output):
        b,n,x,c = depth_output.shape

        if self.config.pool_video:
            depth_output = torch.cat([depth_output[:,:,0:1], depth_output[:,:,1:].mean(2, keepdim=True)], dim=2)
        if n!= self.depth_frame_embedding.shape[1]: #### testing and interpolate
            depth_frame_embedding = F.interpolate(self.depth_frame_embedding.permute(0,2,1),n,mode='nearest').permute(0,2,1)
        else:
            depth_frame_embedding = self.depth_frame_embedding
        depth_output = self.hidden_trans_depth_multimodal(depth_output)
        depth_output =  depth_output + depth_frame_embedding.unsqueeze(-2)
        depth_output = depth_output.reshape(b,-1,self.multimodal_dim)
        depth_output = depth_output + self.depth_type_embeddings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        return depth_output

    def get_multimodal_forward_input_subtitle(self, subtitle_output):
        subtitle_output = self.hidden_trans_subtitle_multimodal(subtitle_output)
        subtitle_output = subtitle_output + self.subtitle_type_embeddings    
        return subtitle_output

    def modify_checkpoint(self, checkpoint):
        new_ckpt = {}
        for k,v in checkpoint.items():
            if 'video' in k:
                new_ckpt[k.replace('video','vision')]=v
            elif 'evaclip_model' in k:
                new_ckpt[k.replace('evaclip_model','vision_encoder')]=v
            elif 'clip_model' in k:    
                new_ckpt[k.replace('clip_model','vision_encoder')]=v
            else:
                new_ckpt[k] = v.float()
        
        checkpoint = new_ckpt
    
        if self.config.frame_embedding_type == 'adaptive':

            if 'vision_frame_embedding' in checkpoint:
                pretrain_embed = checkpoint['vision_frame_embedding']
                if pretrain_embed.shape[1]!=self.config.max_vision_sample_num:
                    pretrain_embed = F.interpolate(pretrain_embed.permute(0,2,1),self.config.max_vision_sample_num,mode='nearest').permute(0,2,1)
                    checkpoint['vision_frame_embedding'] = pretrain_embed
            else: 
                pretrain_embed = checkpoint['vision_perceiver.vision_frame_embedding']
                if pretrain_embed.shape[1]!=self.config.max_vision_sample_num:
                    pretrain_embed = F.interpolate(pretrain_embed.permute(0,2,1),self.config.max_vision_sample_num,mode='nearest').permute(0,2,1)
                    checkpoint['vision_perceiver.vision_frame_embedding'] = pretrain_embed

            if 'audio_frame_embedding' in checkpoint:
                pretrain_embed_a = checkpoint['audio_frame_embedding']
                if pretrain_embed_a.shape[1]!=self.config.max_audio_sample_num:
                    pretrain_embed_a = F.interpolate(pretrain_embed_a.permute(0,2,1),self.config.max_audio_sample_num,mode='nearest').permute(0,2,1)
                    checkpoint['audio_frame_embedding'] = pretrain_embed_a

        if self.config.vision_encoder_type.startswith('clip'):
            vision_width = checkpoint["vision_encoder.visual.positional_embedding"].shape[1]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = checkpoint["vision_encoder.visual.conv1.weight"].shape[-1]
            
            grid_size = round((checkpoint["vision_encoder.visual.positional_embedding"].shape[0] - 1) ** 0.5)
       
            src  = checkpoint["vision_encoder.visual.positional_embedding"]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size!=grid_size:
                src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
                tgt = torch.cat((src_cls,src_oth),dim=0)
                checkpoint["vision_encoder.visual.positional_embedding"] = tgt

        elif self.config.vision_encoder_type.startswith('evaclip'):

            vision_width = checkpoint["vision_encoder.visual.pos_embed"].shape[2]
            vision_layers = len([k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

            vision_patch_size = checkpoint["vision_encoder.visual.patch_embed.proj.weight"].shape[-1]
            
            grid_size = round((checkpoint["vision_encoder.visual.pos_embed"].shape[1] - 1) ** 0.5)
     
            src  = checkpoint["vision_encoder.visual.pos_embed"][0]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size!=grid_size:
                src_oth = F.interpolate(src_oth.reshape(grid_size,grid_size,vision_width).permute(2,0,1).unsqueeze(0),(new_grid_size,new_grid_size),mode='bilinear')
                src_oth = src_oth[0].permute(1,2,0).reshape(-1,src.shape[-1])
                tgt = torch.cat((src_cls,src_oth),dim=0)
                checkpoint["vision_encoder.visual.pos_embed"] = tgt.unsqueeze(0)
        else:
            pass

        return checkpoint

    def load_clip_model(self):
        if self.config.vision_encoder_type.startswith('evaclip'):
            from .evaclip import create_model
            if  self.config.vision_encoder_type == 'evaclip02_base':
                model_name = "EVA02-CLIP-B-16" 
                # pretrained = "./pretrained_weights/clip/EVA02_CLIP_B_psz16_s8B.pt" 
                self.vision_dim = 768
                
            elif self.config.vision_encoder_type == 'evaclip02_base_self':
                model_name = "EVA02-CLIP-B-16" 
                # pretrained = "./pretrained_weights/clip/EVA02_B_psz14to16.pt"
                self.vision_dim = 768

            elif self.config.vision_encoder_type == 'evaclip02_large':
                model_name = "EVA02-CLIP-L-14" 
                # pretrained = "./pretrained_weights/clip/EVA02_CLIP_L_psz14_s4B.pt"
                self.vision_dim = 1024

            elif self.config.vision_encoder_type == 'evaclip02_bige':
                model_name = "EVA02-CLIP-bigE-14-plus" 
                # pretrained = "./pretrained_weights/clip/EVA02_CLIP_E_psz14_plus_s9B.pt" 
                self.vision_dim = 1792

            elif self.config.vision_encoder_type == 'evaclip01_giant':
                model_name = "EVA01-CLIP-g-14" 
                # pretrained = "./pretrained_weights/clip/EVA01_CLIP_g_14_psz14_s11B.pt"
                self.vision_dim = 1408
            

            self.vision_encoder = create_model(model_name, force_custom_clip=True, image_size = self.config.vision_resolution)
            
            if self.config.checkpointing:
                self.vision_encoder.set_grad_checkpointing()
        
        else:
            ### openai clip

            from .clip.clip import build_model
            from .clip.clip import Transformer
            if  self.config.vision_encoder_type == 'clip_vit_base_16':
                # 512 224 12 768 16 77 49408 512 8 12 0 False
                clip_weight = torch.jit.load('/public/hdli/VLPPP/pretrained_weights/clip/ViT-B-16.pt', map_location='cpu')
                self.vision_dim = 768
            elif self.config.vision_encoder_type == 'clip_vit_large_14_336px':
                clip_weight = torch.jit.load('/public/hdli/VLPPP/pretrained_weights/clip/ViT-L-14-336px.pt', map_location='cpu')
                self.vision_dim = 1024
            clip_weight = clip_weight.state_dict()

            self.vision_encoder = build_model(clip_weight, self.config.vision_resolution, self.config.checkpointing).float()


class MiCo(MMGeneralModule):
    """ VLP pretraining """
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.construct_vision_encoder()
        self.construct_audio_encoder()
        self.construct_depth_encoder()
        self.construct_multimodal_encoder()

        contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_s = Contra_head(self.multimodal_dim, contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, contra_dim)
        self.contra_head_d = Contra_head(self.depth_dim, contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, contra_dim)
        self.contra_head_id = nn.Linear(self.vision_dim + self.depth_dim, contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.depth_frame_embedding = nn.Parameter(0.02 * torch.randn(1, self.config.max_depth_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_depth_multimodal = nn.Sequential(nn.Linear(self.depth_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.depth_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim)) 
        self.beam_size  = config.beam_size
        self.itm_ratio = config.itm_ratio   
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len


    @classmethod
    def from_pretrained(cls, opts, state_dict, *inputs, **kwargs):
        model = cls(opts, *inputs, **kwargs)
        missing_keys,unexpected_keys = model.load_state_dict(state_dict,strict=False)
        del model.vision_encoder.text
        if state_dict != {}:
            print(f"Unexpected keys {unexpected_keys}")
            print(f"missing_keys  {missing_keys}")
        return model
