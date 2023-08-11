class GraphModule(torch.nn.Module):
    def forward(self, sample : torch.Tensor, timestep : torch.Tensor, encoder_hidden_states : torch.Tensor):
        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:793, code: timesteps = timesteps[None].to(sample.device)
        getitem = timestep[None];  timestep = None
        to = getitem.to(device(type='cuda', index=0));  getitem = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:796, code: timesteps = timesteps.expand(sample.shape[0])
        expand = to.expand(2);  to = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:43, code: exponent = -math.log(max_period) * torch.arange(
        arange = torch.arange(start = 0, end = 160, dtype = torch.float32, device = device(type='cuda', index=0))
        mul = -9.210340371976184 * arange;  arange = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:46, code: exponent = exponent / (half_dim - downscale_freq_shift)
        truediv = mul / 160;  mul = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:48, code: emb = torch.exp(exponent)
        exp = torch.exp(truediv);  truediv = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:49, code: emb = timesteps[:, None].float() * emb[None, :]
        getitem_1 = expand[(slice(None, None, None), None)];  expand = None
        float_1 = getitem_1.float();  getitem_1 = None
        getitem_2 = exp[(None, slice(None, None, None))];  exp = None
        mul_1 = float_1 * getitem_2;  float_1 = getitem_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:52, code: emb = scale * emb
        mul_2 = 1 * mul_1;  mul_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:55, code: emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        sin = torch.sin(mul_2)
        cos = torch.cos(mul_2);  mul_2 = None
        cat = torch.cat([sin, cos], dim = -1);  sin = cos = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:59, code: emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        getitem_3 = cat[(slice(None, None, None), slice(160, None, None))]
        getitem_4 = cat[(slice(None, None, None), slice(None, 160, None))];  cat = None
        cat_1 = torch.cat([getitem_3, getitem_4], dim = -1);  getitem_3 = getitem_4 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:803, code: t_emb = t_emb.to(dtype=sample.dtype)
        to_1 = cat_1.to(dtype = torch.float16);  cat_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:192, code: sample = self.linear_1(sample)
        self_time_embedding_linear_1 = self.self_time_embedding_linear_1(to_1);  to_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:195, code: sample = self.act(sample)
        self_time_embedding_act = self.self_time_embedding_act(self_time_embedding_linear_1);  self_time_embedding_linear_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/embeddings.py:197, code: sample = self.linear_2(sample)
        self_time_embedding_linear_2 = self.self_time_embedding_linear_2(self_time_embedding_act);  self_time_embedding_act = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:900, code: sample = self.conv_in(sample)
        self_conv_in = self.self_conv_in(sample);  sample = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_0_resnets_0_norm1 = self.self_down_blocks_0_resnets_0_norm1(self_conv_in)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_0_resnets_0_nonlinearity = self.self_down_blocks_0_resnets_0_nonlinearity(self_down_blocks_0_resnets_0_norm1);  self_down_blocks_0_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_resnets_0_conv1_weight = self.self_down_blocks_0_resnets_0_conv1_weight
        self_down_blocks_0_resnets_0_conv1_bias = self.self_down_blocks_0_resnets_0_conv1_bias
        conv2d = torch.conv2d(self_down_blocks_0_resnets_0_nonlinearity, self_down_blocks_0_resnets_0_conv1_weight, self_down_blocks_0_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_0_resnets_0_nonlinearity = self_down_blocks_0_resnets_0_conv1_weight = self_down_blocks_0_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_0_resnets_0_nonlinearity_1 = self.self_down_blocks_0_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_0_resnets_0_time_emb_proj_weight = self.self_down_blocks_0_resnets_0_time_emb_proj_weight
        self_down_blocks_0_resnets_0_time_emb_proj_bias = self.self_down_blocks_0_resnets_0_time_emb_proj_bias
        linear = torch._C._nn.linear(self_down_blocks_0_resnets_0_nonlinearity_1, self_down_blocks_0_resnets_0_time_emb_proj_weight, self_down_blocks_0_resnets_0_time_emb_proj_bias);  self_down_blocks_0_resnets_0_nonlinearity_1 = self_down_blocks_0_resnets_0_time_emb_proj_weight = self_down_blocks_0_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_5 = linear[(slice(None, None, None), slice(None, None, None), None, None)];  linear = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add = conv2d + getitem_5;  conv2d = getitem_5 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_0_resnets_0_norm2 = self.self_down_blocks_0_resnets_0_norm2(add);  add = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_0_resnets_0_nonlinearity_2 = self.self_down_blocks_0_resnets_0_nonlinearity(self_down_blocks_0_resnets_0_norm2);  self_down_blocks_0_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_0_resnets_0_dropout = self.self_down_blocks_0_resnets_0_dropout(self_down_blocks_0_resnets_0_nonlinearity_2);  self_down_blocks_0_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_resnets_0_conv2_weight = self.self_down_blocks_0_resnets_0_conv2_weight
        self_down_blocks_0_resnets_0_conv2_bias = self.self_down_blocks_0_resnets_0_conv2_bias
        conv2d_1 = torch.conv2d(self_down_blocks_0_resnets_0_dropout, self_down_blocks_0_resnets_0_conv2_weight, self_down_blocks_0_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_0_resnets_0_dropout = self_down_blocks_0_resnets_0_conv2_weight = self_down_blocks_0_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_1 = self_conv_in + conv2d_1;  conv2d_1 = None
        truediv_1 = add_1 / 1.0;  add_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_down_blocks_0_attentions_0_norm = self.self_down_blocks_0_attentions_0_norm(truediv_1)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_attentions_0_proj_in_weight = self.self_down_blocks_0_attentions_0_proj_in_weight
        self_down_blocks_0_attentions_0_proj_in_bias = self.self_down_blocks_0_attentions_0_proj_in_bias
        conv2d_2 = torch.conv2d(self_down_blocks_0_attentions_0_norm, self_down_blocks_0_attentions_0_proj_in_weight, self_down_blocks_0_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_0_attentions_0_norm = self_down_blocks_0_attentions_0_proj_in_weight = self_down_blocks_0_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute = conv2d_2.permute(0, 2, 3, 1);  conv2d_2 = None
        reshape = permute.reshape(2, 4096, 320);  permute = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_down_blocks_0_attentions_0_transformer_blocks_0 = self.self_down_blocks_0_attentions_0_transformer_blocks_0(reshape, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_1 = self_down_blocks_0_attentions_0_transformer_blocks_0.reshape(2, 64, 64, 320);  self_down_blocks_0_attentions_0_transformer_blocks_0 = None
        permute_1 = reshape_1.permute(0, 3, 1, 2);  reshape_1 = None
        contiguous = permute_1.contiguous();  permute_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_attentions_0_proj_out_weight = self.self_down_blocks_0_attentions_0_proj_out_weight
        self_down_blocks_0_attentions_0_proj_out_bias = self.self_down_blocks_0_attentions_0_proj_out_bias
        conv2d_3 = torch.conv2d(contiguous, self_down_blocks_0_attentions_0_proj_out_weight, self_down_blocks_0_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous = self_down_blocks_0_attentions_0_proj_out_weight = self_down_blocks_0_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_2 = conv2d_3 + truediv_1;  conv2d_3 = truediv_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_0_resnets_1_norm1 = self.self_down_blocks_0_resnets_1_norm1(add_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_0_resnets_1_nonlinearity = self.self_down_blocks_0_resnets_1_nonlinearity(self_down_blocks_0_resnets_1_norm1);  self_down_blocks_0_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_resnets_1_conv1_weight = self.self_down_blocks_0_resnets_1_conv1_weight
        self_down_blocks_0_resnets_1_conv1_bias = self.self_down_blocks_0_resnets_1_conv1_bias
        conv2d_4 = torch.conv2d(self_down_blocks_0_resnets_1_nonlinearity, self_down_blocks_0_resnets_1_conv1_weight, self_down_blocks_0_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_0_resnets_1_nonlinearity = self_down_blocks_0_resnets_1_conv1_weight = self_down_blocks_0_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_0_resnets_1_nonlinearity_1 = self.self_down_blocks_0_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_0_resnets_1_time_emb_proj_weight = self.self_down_blocks_0_resnets_1_time_emb_proj_weight
        self_down_blocks_0_resnets_1_time_emb_proj_bias = self.self_down_blocks_0_resnets_1_time_emb_proj_bias
        linear_1 = torch._C._nn.linear(self_down_blocks_0_resnets_1_nonlinearity_1, self_down_blocks_0_resnets_1_time_emb_proj_weight, self_down_blocks_0_resnets_1_time_emb_proj_bias);  self_down_blocks_0_resnets_1_nonlinearity_1 = self_down_blocks_0_resnets_1_time_emb_proj_weight = self_down_blocks_0_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_6 = linear_1[(slice(None, None, None), slice(None, None, None), None, None)];  linear_1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_3 = conv2d_4 + getitem_6;  conv2d_4 = getitem_6 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_0_resnets_1_norm2 = self.self_down_blocks_0_resnets_1_norm2(add_3);  add_3 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_0_resnets_1_nonlinearity_2 = self.self_down_blocks_0_resnets_1_nonlinearity(self_down_blocks_0_resnets_1_norm2);  self_down_blocks_0_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_0_resnets_1_dropout = self.self_down_blocks_0_resnets_1_dropout(self_down_blocks_0_resnets_1_nonlinearity_2);  self_down_blocks_0_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_resnets_1_conv2_weight = self.self_down_blocks_0_resnets_1_conv2_weight
        self_down_blocks_0_resnets_1_conv2_bias = self.self_down_blocks_0_resnets_1_conv2_bias
        conv2d_5 = torch.conv2d(self_down_blocks_0_resnets_1_dropout, self_down_blocks_0_resnets_1_conv2_weight, self_down_blocks_0_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_0_resnets_1_dropout = self_down_blocks_0_resnets_1_conv2_weight = self_down_blocks_0_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_4 = add_2 + conv2d_5;  conv2d_5 = None
        truediv_2 = add_4 / 1.0;  add_4 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_down_blocks_0_attentions_1_norm = self.self_down_blocks_0_attentions_1_norm(truediv_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_attentions_1_proj_in_weight = self.self_down_blocks_0_attentions_1_proj_in_weight
        self_down_blocks_0_attentions_1_proj_in_bias = self.self_down_blocks_0_attentions_1_proj_in_bias
        conv2d_6 = torch.conv2d(self_down_blocks_0_attentions_1_norm, self_down_blocks_0_attentions_1_proj_in_weight, self_down_blocks_0_attentions_1_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_0_attentions_1_norm = self_down_blocks_0_attentions_1_proj_in_weight = self_down_blocks_0_attentions_1_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_2 = conv2d_6.permute(0, 2, 3, 1);  conv2d_6 = None
        reshape_2 = permute_2.reshape(2, 4096, 320);  permute_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_down_blocks_0_attentions_1_transformer_blocks_0 = self.self_down_blocks_0_attentions_1_transformer_blocks_0(reshape_2, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_3 = self_down_blocks_0_attentions_1_transformer_blocks_0.reshape(2, 64, 64, 320);  self_down_blocks_0_attentions_1_transformer_blocks_0 = None
        permute_3 = reshape_3.permute(0, 3, 1, 2);  reshape_3 = None
        contiguous_1 = permute_3.contiguous();  permute_3 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_attentions_1_proj_out_weight = self.self_down_blocks_0_attentions_1_proj_out_weight
        self_down_blocks_0_attentions_1_proj_out_bias = self.self_down_blocks_0_attentions_1_proj_out_bias
        conv2d_7 = torch.conv2d(contiguous_1, self_down_blocks_0_attentions_1_proj_out_weight, self_down_blocks_0_attentions_1_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_1 = self_down_blocks_0_attentions_1_proj_out_weight = self_down_blocks_0_attentions_1_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_5 = conv2d_7 + truediv_2;  conv2d_7 = truediv_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_0_downsamplers_0_conv_weight = self.self_down_blocks_0_downsamplers_0_conv_weight
        self_down_blocks_0_downsamplers_0_conv_bias = self.self_down_blocks_0_downsamplers_0_conv_bias
        conv2d_8 = torch.conv2d(add_5, self_down_blocks_0_downsamplers_0_conv_weight, self_down_blocks_0_downsamplers_0_conv_bias, (2, 2), (1, 1), (1, 1), 1);  self_down_blocks_0_downsamplers_0_conv_weight = self_down_blocks_0_downsamplers_0_conv_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_1_resnets_0_norm1 = self.self_down_blocks_1_resnets_0_norm1(conv2d_8)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_1_resnets_0_nonlinearity = self.self_down_blocks_1_resnets_0_nonlinearity(self_down_blocks_1_resnets_0_norm1);  self_down_blocks_1_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_resnets_0_conv1_weight = self.self_down_blocks_1_resnets_0_conv1_weight
        self_down_blocks_1_resnets_0_conv1_bias = self.self_down_blocks_1_resnets_0_conv1_bias
        conv2d_9 = torch.conv2d(self_down_blocks_1_resnets_0_nonlinearity, self_down_blocks_1_resnets_0_conv1_weight, self_down_blocks_1_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_1_resnets_0_nonlinearity = self_down_blocks_1_resnets_0_conv1_weight = self_down_blocks_1_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_1_resnets_0_nonlinearity_1 = self.self_down_blocks_1_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_1_resnets_0_time_emb_proj_weight = self.self_down_blocks_1_resnets_0_time_emb_proj_weight
        self_down_blocks_1_resnets_0_time_emb_proj_bias = self.self_down_blocks_1_resnets_0_time_emb_proj_bias
        linear_2 = torch._C._nn.linear(self_down_blocks_1_resnets_0_nonlinearity_1, self_down_blocks_1_resnets_0_time_emb_proj_weight, self_down_blocks_1_resnets_0_time_emb_proj_bias);  self_down_blocks_1_resnets_0_nonlinearity_1 = self_down_blocks_1_resnets_0_time_emb_proj_weight = self_down_blocks_1_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_7 = linear_2[(slice(None, None, None), slice(None, None, None), None, None)];  linear_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_6 = conv2d_9 + getitem_7;  conv2d_9 = getitem_7 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_1_resnets_0_norm2 = self.self_down_blocks_1_resnets_0_norm2(add_6);  add_6 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_1_resnets_0_nonlinearity_2 = self.self_down_blocks_1_resnets_0_nonlinearity(self_down_blocks_1_resnets_0_norm2);  self_down_blocks_1_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_1_resnets_0_dropout = self.self_down_blocks_1_resnets_0_dropout(self_down_blocks_1_resnets_0_nonlinearity_2);  self_down_blocks_1_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_resnets_0_conv2_weight = self.self_down_blocks_1_resnets_0_conv2_weight
        self_down_blocks_1_resnets_0_conv2_bias = self.self_down_blocks_1_resnets_0_conv2_bias
        conv2d_10 = torch.conv2d(self_down_blocks_1_resnets_0_dropout, self_down_blocks_1_resnets_0_conv2_weight, self_down_blocks_1_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_1_resnets_0_dropout = self_down_blocks_1_resnets_0_conv2_weight = self_down_blocks_1_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_resnets_0_conv_shortcut_weight = self.self_down_blocks_1_resnets_0_conv_shortcut_weight
        self_down_blocks_1_resnets_0_conv_shortcut_bias = self.self_down_blocks_1_resnets_0_conv_shortcut_bias
        conv2d_11 = torch.conv2d(conv2d_8, self_down_blocks_1_resnets_0_conv_shortcut_weight, self_down_blocks_1_resnets_0_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_1_resnets_0_conv_shortcut_weight = self_down_blocks_1_resnets_0_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_7 = conv2d_11 + conv2d_10;  conv2d_11 = conv2d_10 = None
        truediv_3 = add_7 / 1.0;  add_7 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_down_blocks_1_attentions_0_norm = self.self_down_blocks_1_attentions_0_norm(truediv_3)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_attentions_0_proj_in_weight = self.self_down_blocks_1_attentions_0_proj_in_weight
        self_down_blocks_1_attentions_0_proj_in_bias = self.self_down_blocks_1_attentions_0_proj_in_bias
        conv2d_12 = torch.conv2d(self_down_blocks_1_attentions_0_norm, self_down_blocks_1_attentions_0_proj_in_weight, self_down_blocks_1_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_1_attentions_0_norm = self_down_blocks_1_attentions_0_proj_in_weight = self_down_blocks_1_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_4 = conv2d_12.permute(0, 2, 3, 1);  conv2d_12 = None
        reshape_4 = permute_4.reshape(2, 1024, 640);  permute_4 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_down_blocks_1_attentions_0_transformer_blocks_0 = self.self_down_blocks_1_attentions_0_transformer_blocks_0(reshape_4, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_4 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_5 = self_down_blocks_1_attentions_0_transformer_blocks_0.reshape(2, 32, 32, 640);  self_down_blocks_1_attentions_0_transformer_blocks_0 = None
        permute_5 = reshape_5.permute(0, 3, 1, 2);  reshape_5 = None
        contiguous_2 = permute_5.contiguous();  permute_5 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_attentions_0_proj_out_weight = self.self_down_blocks_1_attentions_0_proj_out_weight
        self_down_blocks_1_attentions_0_proj_out_bias = self.self_down_blocks_1_attentions_0_proj_out_bias
        conv2d_13 = torch.conv2d(contiguous_2, self_down_blocks_1_attentions_0_proj_out_weight, self_down_blocks_1_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_2 = self_down_blocks_1_attentions_0_proj_out_weight = self_down_blocks_1_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_8 = conv2d_13 + truediv_3;  conv2d_13 = truediv_3 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_1_resnets_1_norm1 = self.self_down_blocks_1_resnets_1_norm1(add_8)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_1_resnets_1_nonlinearity = self.self_down_blocks_1_resnets_1_nonlinearity(self_down_blocks_1_resnets_1_norm1);  self_down_blocks_1_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_resnets_1_conv1_weight = self.self_down_blocks_1_resnets_1_conv1_weight
        self_down_blocks_1_resnets_1_conv1_bias = self.self_down_blocks_1_resnets_1_conv1_bias
        conv2d_14 = torch.conv2d(self_down_blocks_1_resnets_1_nonlinearity, self_down_blocks_1_resnets_1_conv1_weight, self_down_blocks_1_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_1_resnets_1_nonlinearity = self_down_blocks_1_resnets_1_conv1_weight = self_down_blocks_1_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_1_resnets_1_nonlinearity_1 = self.self_down_blocks_1_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_1_resnets_1_time_emb_proj_weight = self.self_down_blocks_1_resnets_1_time_emb_proj_weight
        self_down_blocks_1_resnets_1_time_emb_proj_bias = self.self_down_blocks_1_resnets_1_time_emb_proj_bias
        linear_3 = torch._C._nn.linear(self_down_blocks_1_resnets_1_nonlinearity_1, self_down_blocks_1_resnets_1_time_emb_proj_weight, self_down_blocks_1_resnets_1_time_emb_proj_bias);  self_down_blocks_1_resnets_1_nonlinearity_1 = self_down_blocks_1_resnets_1_time_emb_proj_weight = self_down_blocks_1_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_8 = linear_3[(slice(None, None, None), slice(None, None, None), None, None)];  linear_3 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_9 = conv2d_14 + getitem_8;  conv2d_14 = getitem_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_1_resnets_1_norm2 = self.self_down_blocks_1_resnets_1_norm2(add_9);  add_9 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_1_resnets_1_nonlinearity_2 = self.self_down_blocks_1_resnets_1_nonlinearity(self_down_blocks_1_resnets_1_norm2);  self_down_blocks_1_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_1_resnets_1_dropout = self.self_down_blocks_1_resnets_1_dropout(self_down_blocks_1_resnets_1_nonlinearity_2);  self_down_blocks_1_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_resnets_1_conv2_weight = self.self_down_blocks_1_resnets_1_conv2_weight
        self_down_blocks_1_resnets_1_conv2_bias = self.self_down_blocks_1_resnets_1_conv2_bias
        conv2d_15 = torch.conv2d(self_down_blocks_1_resnets_1_dropout, self_down_blocks_1_resnets_1_conv2_weight, self_down_blocks_1_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_1_resnets_1_dropout = self_down_blocks_1_resnets_1_conv2_weight = self_down_blocks_1_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_10 = add_8 + conv2d_15;  conv2d_15 = None
        truediv_4 = add_10 / 1.0;  add_10 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_down_blocks_1_attentions_1_norm = self.self_down_blocks_1_attentions_1_norm(truediv_4)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_attentions_1_proj_in_weight = self.self_down_blocks_1_attentions_1_proj_in_weight
        self_down_blocks_1_attentions_1_proj_in_bias = self.self_down_blocks_1_attentions_1_proj_in_bias
        conv2d_16 = torch.conv2d(self_down_blocks_1_attentions_1_norm, self_down_blocks_1_attentions_1_proj_in_weight, self_down_blocks_1_attentions_1_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_1_attentions_1_norm = self_down_blocks_1_attentions_1_proj_in_weight = self_down_blocks_1_attentions_1_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_6 = conv2d_16.permute(0, 2, 3, 1);  conv2d_16 = None
        reshape_6 = permute_6.reshape(2, 1024, 640);  permute_6 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_down_blocks_1_attentions_1_transformer_blocks_0 = self.self_down_blocks_1_attentions_1_transformer_blocks_0(reshape_6, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_6 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_7 = self_down_blocks_1_attentions_1_transformer_blocks_0.reshape(2, 32, 32, 640);  self_down_blocks_1_attentions_1_transformer_blocks_0 = None
        permute_7 = reshape_7.permute(0, 3, 1, 2);  reshape_7 = None
        contiguous_3 = permute_7.contiguous();  permute_7 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_attentions_1_proj_out_weight = self.self_down_blocks_1_attentions_1_proj_out_weight
        self_down_blocks_1_attentions_1_proj_out_bias = self.self_down_blocks_1_attentions_1_proj_out_bias
        conv2d_17 = torch.conv2d(contiguous_3, self_down_blocks_1_attentions_1_proj_out_weight, self_down_blocks_1_attentions_1_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_3 = self_down_blocks_1_attentions_1_proj_out_weight = self_down_blocks_1_attentions_1_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_11 = conv2d_17 + truediv_4;  conv2d_17 = truediv_4 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_1_downsamplers_0_conv_weight = self.self_down_blocks_1_downsamplers_0_conv_weight
        self_down_blocks_1_downsamplers_0_conv_bias = self.self_down_blocks_1_downsamplers_0_conv_bias
        conv2d_18 = torch.conv2d(add_11, self_down_blocks_1_downsamplers_0_conv_weight, self_down_blocks_1_downsamplers_0_conv_bias, (2, 2), (1, 1), (1, 1), 1);  self_down_blocks_1_downsamplers_0_conv_weight = self_down_blocks_1_downsamplers_0_conv_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_2_resnets_0_norm1 = self.self_down_blocks_2_resnets_0_norm1(conv2d_18)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_2_resnets_0_nonlinearity = self.self_down_blocks_2_resnets_0_nonlinearity(self_down_blocks_2_resnets_0_norm1);  self_down_blocks_2_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_resnets_0_conv1_weight = self.self_down_blocks_2_resnets_0_conv1_weight
        self_down_blocks_2_resnets_0_conv1_bias = self.self_down_blocks_2_resnets_0_conv1_bias
        conv2d_19 = torch.conv2d(self_down_blocks_2_resnets_0_nonlinearity, self_down_blocks_2_resnets_0_conv1_weight, self_down_blocks_2_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_2_resnets_0_nonlinearity = self_down_blocks_2_resnets_0_conv1_weight = self_down_blocks_2_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_2_resnets_0_nonlinearity_1 = self.self_down_blocks_2_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_2_resnets_0_time_emb_proj_weight = self.self_down_blocks_2_resnets_0_time_emb_proj_weight
        self_down_blocks_2_resnets_0_time_emb_proj_bias = self.self_down_blocks_2_resnets_0_time_emb_proj_bias
        linear_4 = torch._C._nn.linear(self_down_blocks_2_resnets_0_nonlinearity_1, self_down_blocks_2_resnets_0_time_emb_proj_weight, self_down_blocks_2_resnets_0_time_emb_proj_bias);  self_down_blocks_2_resnets_0_nonlinearity_1 = self_down_blocks_2_resnets_0_time_emb_proj_weight = self_down_blocks_2_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_9 = linear_4[(slice(None, None, None), slice(None, None, None), None, None)];  linear_4 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_12 = conv2d_19 + getitem_9;  conv2d_19 = getitem_9 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_2_resnets_0_norm2 = self.self_down_blocks_2_resnets_0_norm2(add_12);  add_12 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_2_resnets_0_nonlinearity_2 = self.self_down_blocks_2_resnets_0_nonlinearity(self_down_blocks_2_resnets_0_norm2);  self_down_blocks_2_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_2_resnets_0_dropout = self.self_down_blocks_2_resnets_0_dropout(self_down_blocks_2_resnets_0_nonlinearity_2);  self_down_blocks_2_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_resnets_0_conv2_weight = self.self_down_blocks_2_resnets_0_conv2_weight
        self_down_blocks_2_resnets_0_conv2_bias = self.self_down_blocks_2_resnets_0_conv2_bias
        conv2d_20 = torch.conv2d(self_down_blocks_2_resnets_0_dropout, self_down_blocks_2_resnets_0_conv2_weight, self_down_blocks_2_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_2_resnets_0_dropout = self_down_blocks_2_resnets_0_conv2_weight = self_down_blocks_2_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_resnets_0_conv_shortcut_weight = self.self_down_blocks_2_resnets_0_conv_shortcut_weight
        self_down_blocks_2_resnets_0_conv_shortcut_bias = self.self_down_blocks_2_resnets_0_conv_shortcut_bias
        conv2d_21 = torch.conv2d(conv2d_18, self_down_blocks_2_resnets_0_conv_shortcut_weight, self_down_blocks_2_resnets_0_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_2_resnets_0_conv_shortcut_weight = self_down_blocks_2_resnets_0_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_13 = conv2d_21 + conv2d_20;  conv2d_21 = conv2d_20 = None
        truediv_5 = add_13 / 1.0;  add_13 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_down_blocks_2_attentions_0_norm = self.self_down_blocks_2_attentions_0_norm(truediv_5)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_attentions_0_proj_in_weight = self.self_down_blocks_2_attentions_0_proj_in_weight
        self_down_blocks_2_attentions_0_proj_in_bias = self.self_down_blocks_2_attentions_0_proj_in_bias
        conv2d_22 = torch.conv2d(self_down_blocks_2_attentions_0_norm, self_down_blocks_2_attentions_0_proj_in_weight, self_down_blocks_2_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_2_attentions_0_norm = self_down_blocks_2_attentions_0_proj_in_weight = self_down_blocks_2_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_8 = conv2d_22.permute(0, 2, 3, 1);  conv2d_22 = None
        reshape_8 = permute_8.reshape(2, 256, 1280);  permute_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_down_blocks_2_attentions_0_transformer_blocks_0 = self.self_down_blocks_2_attentions_0_transformer_blocks_0(reshape_8, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_9 = self_down_blocks_2_attentions_0_transformer_blocks_0.reshape(2, 16, 16, 1280);  self_down_blocks_2_attentions_0_transformer_blocks_0 = None
        permute_9 = reshape_9.permute(0, 3, 1, 2);  reshape_9 = None
        contiguous_4 = permute_9.contiguous();  permute_9 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_attentions_0_proj_out_weight = self.self_down_blocks_2_attentions_0_proj_out_weight
        self_down_blocks_2_attentions_0_proj_out_bias = self.self_down_blocks_2_attentions_0_proj_out_bias
        conv2d_23 = torch.conv2d(contiguous_4, self_down_blocks_2_attentions_0_proj_out_weight, self_down_blocks_2_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_4 = self_down_blocks_2_attentions_0_proj_out_weight = self_down_blocks_2_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_14 = conv2d_23 + truediv_5;  conv2d_23 = truediv_5 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_2_resnets_1_norm1 = self.self_down_blocks_2_resnets_1_norm1(add_14)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_2_resnets_1_nonlinearity = self.self_down_blocks_2_resnets_1_nonlinearity(self_down_blocks_2_resnets_1_norm1);  self_down_blocks_2_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_resnets_1_conv1_weight = self.self_down_blocks_2_resnets_1_conv1_weight
        self_down_blocks_2_resnets_1_conv1_bias = self.self_down_blocks_2_resnets_1_conv1_bias
        conv2d_24 = torch.conv2d(self_down_blocks_2_resnets_1_nonlinearity, self_down_blocks_2_resnets_1_conv1_weight, self_down_blocks_2_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_2_resnets_1_nonlinearity = self_down_blocks_2_resnets_1_conv1_weight = self_down_blocks_2_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_2_resnets_1_nonlinearity_1 = self.self_down_blocks_2_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_2_resnets_1_time_emb_proj_weight = self.self_down_blocks_2_resnets_1_time_emb_proj_weight
        self_down_blocks_2_resnets_1_time_emb_proj_bias = self.self_down_blocks_2_resnets_1_time_emb_proj_bias
        linear_5 = torch._C._nn.linear(self_down_blocks_2_resnets_1_nonlinearity_1, self_down_blocks_2_resnets_1_time_emb_proj_weight, self_down_blocks_2_resnets_1_time_emb_proj_bias);  self_down_blocks_2_resnets_1_nonlinearity_1 = self_down_blocks_2_resnets_1_time_emb_proj_weight = self_down_blocks_2_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_10 = linear_5[(slice(None, None, None), slice(None, None, None), None, None)];  linear_5 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_15 = conv2d_24 + getitem_10;  conv2d_24 = getitem_10 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_2_resnets_1_norm2 = self.self_down_blocks_2_resnets_1_norm2(add_15);  add_15 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_2_resnets_1_nonlinearity_2 = self.self_down_blocks_2_resnets_1_nonlinearity(self_down_blocks_2_resnets_1_norm2);  self_down_blocks_2_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_2_resnets_1_dropout = self.self_down_blocks_2_resnets_1_dropout(self_down_blocks_2_resnets_1_nonlinearity_2);  self_down_blocks_2_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_resnets_1_conv2_weight = self.self_down_blocks_2_resnets_1_conv2_weight
        self_down_blocks_2_resnets_1_conv2_bias = self.self_down_blocks_2_resnets_1_conv2_bias
        conv2d_25 = torch.conv2d(self_down_blocks_2_resnets_1_dropout, self_down_blocks_2_resnets_1_conv2_weight, self_down_blocks_2_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_2_resnets_1_dropout = self_down_blocks_2_resnets_1_conv2_weight = self_down_blocks_2_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_16 = add_14 + conv2d_25;  conv2d_25 = None
        truediv_6 = add_16 / 1.0;  add_16 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_down_blocks_2_attentions_1_norm = self.self_down_blocks_2_attentions_1_norm(truediv_6)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_attentions_1_proj_in_weight = self.self_down_blocks_2_attentions_1_proj_in_weight
        self_down_blocks_2_attentions_1_proj_in_bias = self.self_down_blocks_2_attentions_1_proj_in_bias
        conv2d_26 = torch.conv2d(self_down_blocks_2_attentions_1_norm, self_down_blocks_2_attentions_1_proj_in_weight, self_down_blocks_2_attentions_1_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_down_blocks_2_attentions_1_norm = self_down_blocks_2_attentions_1_proj_in_weight = self_down_blocks_2_attentions_1_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_10 = conv2d_26.permute(0, 2, 3, 1);  conv2d_26 = None
        reshape_10 = permute_10.reshape(2, 256, 1280);  permute_10 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_down_blocks_2_attentions_1_transformer_blocks_0 = self.self_down_blocks_2_attentions_1_transformer_blocks_0(reshape_10, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_10 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_11 = self_down_blocks_2_attentions_1_transformer_blocks_0.reshape(2, 16, 16, 1280);  self_down_blocks_2_attentions_1_transformer_blocks_0 = None
        permute_11 = reshape_11.permute(0, 3, 1, 2);  reshape_11 = None
        contiguous_5 = permute_11.contiguous();  permute_11 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_attentions_1_proj_out_weight = self.self_down_blocks_2_attentions_1_proj_out_weight
        self_down_blocks_2_attentions_1_proj_out_bias = self.self_down_blocks_2_attentions_1_proj_out_bias
        conv2d_27 = torch.conv2d(contiguous_5, self_down_blocks_2_attentions_1_proj_out_weight, self_down_blocks_2_attentions_1_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_5 = self_down_blocks_2_attentions_1_proj_out_weight = self_down_blocks_2_attentions_1_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_17 = conv2d_27 + truediv_6;  conv2d_27 = truediv_6 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_2_downsamplers_0_conv_weight = self.self_down_blocks_2_downsamplers_0_conv_weight
        self_down_blocks_2_downsamplers_0_conv_bias = self.self_down_blocks_2_downsamplers_0_conv_bias
        conv2d_28 = torch.conv2d(add_17, self_down_blocks_2_downsamplers_0_conv_weight, self_down_blocks_2_downsamplers_0_conv_bias, (2, 2), (1, 1), (1, 1), 1);  self_down_blocks_2_downsamplers_0_conv_weight = self_down_blocks_2_downsamplers_0_conv_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_3_resnets_0_norm1 = self.self_down_blocks_3_resnets_0_norm1(conv2d_28)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_3_resnets_0_nonlinearity = self.self_down_blocks_3_resnets_0_nonlinearity(self_down_blocks_3_resnets_0_norm1);  self_down_blocks_3_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_3_resnets_0_conv1_weight = self.self_down_blocks_3_resnets_0_conv1_weight
        self_down_blocks_3_resnets_0_conv1_bias = self.self_down_blocks_3_resnets_0_conv1_bias
        conv2d_29 = torch.conv2d(self_down_blocks_3_resnets_0_nonlinearity, self_down_blocks_3_resnets_0_conv1_weight, self_down_blocks_3_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_3_resnets_0_nonlinearity = self_down_blocks_3_resnets_0_conv1_weight = self_down_blocks_3_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_3_resnets_0_nonlinearity_1 = self.self_down_blocks_3_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_3_resnets_0_time_emb_proj_weight = self.self_down_blocks_3_resnets_0_time_emb_proj_weight
        self_down_blocks_3_resnets_0_time_emb_proj_bias = self.self_down_blocks_3_resnets_0_time_emb_proj_bias
        linear_6 = torch._C._nn.linear(self_down_blocks_3_resnets_0_nonlinearity_1, self_down_blocks_3_resnets_0_time_emb_proj_weight, self_down_blocks_3_resnets_0_time_emb_proj_bias);  self_down_blocks_3_resnets_0_nonlinearity_1 = self_down_blocks_3_resnets_0_time_emb_proj_weight = self_down_blocks_3_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_11 = linear_6[(slice(None, None, None), slice(None, None, None), None, None)];  linear_6 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_18 = conv2d_29 + getitem_11;  conv2d_29 = getitem_11 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_3_resnets_0_norm2 = self.self_down_blocks_3_resnets_0_norm2(add_18);  add_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_3_resnets_0_nonlinearity_2 = self.self_down_blocks_3_resnets_0_nonlinearity(self_down_blocks_3_resnets_0_norm2);  self_down_blocks_3_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_3_resnets_0_dropout = self.self_down_blocks_3_resnets_0_dropout(self_down_blocks_3_resnets_0_nonlinearity_2);  self_down_blocks_3_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_3_resnets_0_conv2_weight = self.self_down_blocks_3_resnets_0_conv2_weight
        self_down_blocks_3_resnets_0_conv2_bias = self.self_down_blocks_3_resnets_0_conv2_bias
        conv2d_30 = torch.conv2d(self_down_blocks_3_resnets_0_dropout, self_down_blocks_3_resnets_0_conv2_weight, self_down_blocks_3_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_3_resnets_0_dropout = self_down_blocks_3_resnets_0_conv2_weight = self_down_blocks_3_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_19 = conv2d_28 + conv2d_30;  conv2d_30 = None
        truediv_7 = add_19 / 1.0;  add_19 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_down_blocks_3_resnets_1_norm1 = self.self_down_blocks_3_resnets_1_norm1(truediv_7)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_3_resnets_1_nonlinearity = self.self_down_blocks_3_resnets_1_nonlinearity(self_down_blocks_3_resnets_1_norm1);  self_down_blocks_3_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_3_resnets_1_conv1_weight = self.self_down_blocks_3_resnets_1_conv1_weight
        self_down_blocks_3_resnets_1_conv1_bias = self.self_down_blocks_3_resnets_1_conv1_bias
        conv2d_31 = torch.conv2d(self_down_blocks_3_resnets_1_nonlinearity, self_down_blocks_3_resnets_1_conv1_weight, self_down_blocks_3_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_3_resnets_1_nonlinearity = self_down_blocks_3_resnets_1_conv1_weight = self_down_blocks_3_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_down_blocks_3_resnets_1_nonlinearity_1 = self.self_down_blocks_3_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_down_blocks_3_resnets_1_time_emb_proj_weight = self.self_down_blocks_3_resnets_1_time_emb_proj_weight
        self_down_blocks_3_resnets_1_time_emb_proj_bias = self.self_down_blocks_3_resnets_1_time_emb_proj_bias
        linear_7 = torch._C._nn.linear(self_down_blocks_3_resnets_1_nonlinearity_1, self_down_blocks_3_resnets_1_time_emb_proj_weight, self_down_blocks_3_resnets_1_time_emb_proj_bias);  self_down_blocks_3_resnets_1_nonlinearity_1 = self_down_blocks_3_resnets_1_time_emb_proj_weight = self_down_blocks_3_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_12 = linear_7[(slice(None, None, None), slice(None, None, None), None, None)];  linear_7 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_20 = conv2d_31 + getitem_12;  conv2d_31 = getitem_12 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_down_blocks_3_resnets_1_norm2 = self.self_down_blocks_3_resnets_1_norm2(add_20);  add_20 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_down_blocks_3_resnets_1_nonlinearity_2 = self.self_down_blocks_3_resnets_1_nonlinearity(self_down_blocks_3_resnets_1_norm2);  self_down_blocks_3_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_down_blocks_3_resnets_1_dropout = self.self_down_blocks_3_resnets_1_dropout(self_down_blocks_3_resnets_1_nonlinearity_2);  self_down_blocks_3_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_down_blocks_3_resnets_1_conv2_weight = self.self_down_blocks_3_resnets_1_conv2_weight
        self_down_blocks_3_resnets_1_conv2_bias = self.self_down_blocks_3_resnets_1_conv2_bias
        conv2d_32 = torch.conv2d(self_down_blocks_3_resnets_1_dropout, self_down_blocks_3_resnets_1_conv2_weight, self_down_blocks_3_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_down_blocks_3_resnets_1_dropout = self_down_blocks_3_resnets_1_conv2_weight = self_down_blocks_3_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_21 = truediv_7 + conv2d_32;  conv2d_32 = None
        truediv_8 = add_21 / 1.0;  add_21 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_mid_block_resnets_0_norm1 = self.self_mid_block_resnets_0_norm1(truediv_8)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_mid_block_resnets_0_nonlinearity = self.self_mid_block_resnets_0_nonlinearity(self_mid_block_resnets_0_norm1);  self_mid_block_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_mid_block_resnets_0_conv1_weight = self.self_mid_block_resnets_0_conv1_weight
        self_mid_block_resnets_0_conv1_bias = self.self_mid_block_resnets_0_conv1_bias
        conv2d_33 = torch.conv2d(self_mid_block_resnets_0_nonlinearity, self_mid_block_resnets_0_conv1_weight, self_mid_block_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_mid_block_resnets_0_nonlinearity = self_mid_block_resnets_0_conv1_weight = self_mid_block_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_mid_block_resnets_0_nonlinearity_1 = self.self_mid_block_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_mid_block_resnets_0_time_emb_proj_weight = self.self_mid_block_resnets_0_time_emb_proj_weight
        self_mid_block_resnets_0_time_emb_proj_bias = self.self_mid_block_resnets_0_time_emb_proj_bias
        linear_8 = torch._C._nn.linear(self_mid_block_resnets_0_nonlinearity_1, self_mid_block_resnets_0_time_emb_proj_weight, self_mid_block_resnets_0_time_emb_proj_bias);  self_mid_block_resnets_0_nonlinearity_1 = self_mid_block_resnets_0_time_emb_proj_weight = self_mid_block_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_13 = linear_8[(slice(None, None, None), slice(None, None, None), None, None)];  linear_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_22 = conv2d_33 + getitem_13;  conv2d_33 = getitem_13 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_mid_block_resnets_0_norm2 = self.self_mid_block_resnets_0_norm2(add_22);  add_22 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_mid_block_resnets_0_nonlinearity_2 = self.self_mid_block_resnets_0_nonlinearity(self_mid_block_resnets_0_norm2);  self_mid_block_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_mid_block_resnets_0_dropout = self.self_mid_block_resnets_0_dropout(self_mid_block_resnets_0_nonlinearity_2);  self_mid_block_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_mid_block_resnets_0_conv2_weight = self.self_mid_block_resnets_0_conv2_weight
        self_mid_block_resnets_0_conv2_bias = self.self_mid_block_resnets_0_conv2_bias
        conv2d_34 = torch.conv2d(self_mid_block_resnets_0_dropout, self_mid_block_resnets_0_conv2_weight, self_mid_block_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_mid_block_resnets_0_dropout = self_mid_block_resnets_0_conv2_weight = self_mid_block_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_23 = truediv_8 + conv2d_34;  conv2d_34 = None
        truediv_9 = add_23 / 1;  add_23 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_mid_block_attentions_0_norm = self.self_mid_block_attentions_0_norm(truediv_9)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_mid_block_attentions_0_proj_in_weight = self.self_mid_block_attentions_0_proj_in_weight
        self_mid_block_attentions_0_proj_in_bias = self.self_mid_block_attentions_0_proj_in_bias
        conv2d_35 = torch.conv2d(self_mid_block_attentions_0_norm, self_mid_block_attentions_0_proj_in_weight, self_mid_block_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_mid_block_attentions_0_norm = self_mid_block_attentions_0_proj_in_weight = self_mid_block_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_12 = conv2d_35.permute(0, 2, 3, 1);  conv2d_35 = None
        reshape_12 = permute_12.reshape(2, 64, 1280);  permute_12 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_mid_block_attentions_0_transformer_blocks_0 = self.self_mid_block_attentions_0_transformer_blocks_0(reshape_12, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_12 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_13 = self_mid_block_attentions_0_transformer_blocks_0.reshape(2, 8, 8, 1280);  self_mid_block_attentions_0_transformer_blocks_0 = None
        permute_13 = reshape_13.permute(0, 3, 1, 2);  reshape_13 = None
        contiguous_6 = permute_13.contiguous();  permute_13 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_mid_block_attentions_0_proj_out_weight = self.self_mid_block_attentions_0_proj_out_weight
        self_mid_block_attentions_0_proj_out_bias = self.self_mid_block_attentions_0_proj_out_bias
        conv2d_36 = torch.conv2d(contiguous_6, self_mid_block_attentions_0_proj_out_weight, self_mid_block_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_6 = self_mid_block_attentions_0_proj_out_weight = self_mid_block_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_24 = conv2d_36 + truediv_9;  conv2d_36 = truediv_9 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_mid_block_resnets_slice_1__none__none___0_norm1 = self.self_mid_block_resnets_slice_1__None__None___0_norm1(add_24)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_mid_block_resnets_slice_1__none__none___0_nonlinearity = self.self_mid_block_resnets_slice_1__None__None___0_nonlinearity(self_mid_block_resnets_slice_1__none__none___0_norm1);  self_mid_block_resnets_slice_1__none__none___0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_mid_block_resnets_slice_1__none__none___0_conv1_weight = self.self_mid_block_resnets_slice_1__None__None___0_conv1_weight
        self_mid_block_resnets_slice_1__none__none___0_conv1_bias = self.self_mid_block_resnets_slice_1__None__None___0_conv1_bias
        conv2d_37 = torch.conv2d(self_mid_block_resnets_slice_1__none__none___0_nonlinearity, self_mid_block_resnets_slice_1__none__none___0_conv1_weight, self_mid_block_resnets_slice_1__none__none___0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_mid_block_resnets_slice_1__none__none___0_nonlinearity = self_mid_block_resnets_slice_1__none__none___0_conv1_weight = self_mid_block_resnets_slice_1__none__none___0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_mid_block_resnets_slice_1__none__none___0_nonlinearity_1 = self.self_mid_block_resnets_slice_1__None__None___0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_mid_block_resnets_slice_1__none__none___0_time_emb_proj_weight = self.self_mid_block_resnets_slice_1__None__None___0_time_emb_proj_weight
        self_mid_block_resnets_slice_1__none__none___0_time_emb_proj_bias = self.self_mid_block_resnets_slice_1__None__None___0_time_emb_proj_bias
        linear_9 = torch._C._nn.linear(self_mid_block_resnets_slice_1__none__none___0_nonlinearity_1, self_mid_block_resnets_slice_1__none__none___0_time_emb_proj_weight, self_mid_block_resnets_slice_1__none__none___0_time_emb_proj_bias);  self_mid_block_resnets_slice_1__none__none___0_nonlinearity_1 = self_mid_block_resnets_slice_1__none__none___0_time_emb_proj_weight = self_mid_block_resnets_slice_1__none__none___0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_14 = linear_9[(slice(None, None, None), slice(None, None, None), None, None)];  linear_9 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_25 = conv2d_37 + getitem_14;  conv2d_37 = getitem_14 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_mid_block_resnets_slice_1__none__none___0_norm2 = self.self_mid_block_resnets_slice_1__None__None___0_norm2(add_25);  add_25 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_mid_block_resnets_slice_1__none__none___0_nonlinearity_2 = self.self_mid_block_resnets_slice_1__None__None___0_nonlinearity(self_mid_block_resnets_slice_1__none__none___0_norm2);  self_mid_block_resnets_slice_1__none__none___0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_mid_block_resnets_slice_1__none__none___0_dropout = self.self_mid_block_resnets_slice_1__None__None___0_dropout(self_mid_block_resnets_slice_1__none__none___0_nonlinearity_2);  self_mid_block_resnets_slice_1__none__none___0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_mid_block_resnets_slice_1__none__none___0_conv2_weight = self.self_mid_block_resnets_slice_1__None__None___0_conv2_weight
        self_mid_block_resnets_slice_1__none__none___0_conv2_bias = self.self_mid_block_resnets_slice_1__None__None___0_conv2_bias
        conv2d_38 = torch.conv2d(self_mid_block_resnets_slice_1__none__none___0_dropout, self_mid_block_resnets_slice_1__none__none___0_conv2_weight, self_mid_block_resnets_slice_1__none__none___0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_mid_block_resnets_slice_1__none__none___0_dropout = self_mid_block_resnets_slice_1__none__none___0_conv2_weight = self_mid_block_resnets_slice_1__none__none___0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_26 = add_24 + conv2d_38;  add_24 = conv2d_38 = None
        truediv_10 = add_26 / 1;  add_26 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2203, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_2 = torch.cat([truediv_10, truediv_8], dim = 1);  truediv_10 = truediv_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_0_resnets_0_norm1 = self.self_up_blocks_0_resnets_0_norm1(cat_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_0_resnets_0_nonlinearity = self.self_up_blocks_0_resnets_0_nonlinearity(self_up_blocks_0_resnets_0_norm1);  self_up_blocks_0_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_0_conv1_weight = self.self_up_blocks_0_resnets_0_conv1_weight
        self_up_blocks_0_resnets_0_conv1_bias = self.self_up_blocks_0_resnets_0_conv1_bias
        conv2d_39 = torch.conv2d(self_up_blocks_0_resnets_0_nonlinearity, self_up_blocks_0_resnets_0_conv1_weight, self_up_blocks_0_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_0_resnets_0_nonlinearity = self_up_blocks_0_resnets_0_conv1_weight = self_up_blocks_0_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_0_resnets_0_nonlinearity_1 = self.self_up_blocks_0_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_0_resnets_0_time_emb_proj_weight = self.self_up_blocks_0_resnets_0_time_emb_proj_weight
        self_up_blocks_0_resnets_0_time_emb_proj_bias = self.self_up_blocks_0_resnets_0_time_emb_proj_bias
        linear_10 = torch._C._nn.linear(self_up_blocks_0_resnets_0_nonlinearity_1, self_up_blocks_0_resnets_0_time_emb_proj_weight, self_up_blocks_0_resnets_0_time_emb_proj_bias);  self_up_blocks_0_resnets_0_nonlinearity_1 = self_up_blocks_0_resnets_0_time_emb_proj_weight = self_up_blocks_0_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_15 = linear_10[(slice(None, None, None), slice(None, None, None), None, None)];  linear_10 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_27 = conv2d_39 + getitem_15;  conv2d_39 = getitem_15 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_0_resnets_0_norm2 = self.self_up_blocks_0_resnets_0_norm2(add_27);  add_27 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_0_resnets_0_nonlinearity_2 = self.self_up_blocks_0_resnets_0_nonlinearity(self_up_blocks_0_resnets_0_norm2);  self_up_blocks_0_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_0_resnets_0_dropout = self.self_up_blocks_0_resnets_0_dropout(self_up_blocks_0_resnets_0_nonlinearity_2);  self_up_blocks_0_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_0_conv2_weight = self.self_up_blocks_0_resnets_0_conv2_weight
        self_up_blocks_0_resnets_0_conv2_bias = self.self_up_blocks_0_resnets_0_conv2_bias
        conv2d_40 = torch.conv2d(self_up_blocks_0_resnets_0_dropout, self_up_blocks_0_resnets_0_conv2_weight, self_up_blocks_0_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_0_resnets_0_dropout = self_up_blocks_0_resnets_0_conv2_weight = self_up_blocks_0_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_0_conv_shortcut_weight = self.self_up_blocks_0_resnets_0_conv_shortcut_weight
        self_up_blocks_0_resnets_0_conv_shortcut_bias = self.self_up_blocks_0_resnets_0_conv_shortcut_bias
        conv2d_41 = torch.conv2d(cat_2, self_up_blocks_0_resnets_0_conv_shortcut_weight, self_up_blocks_0_resnets_0_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_2 = self_up_blocks_0_resnets_0_conv_shortcut_weight = self_up_blocks_0_resnets_0_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_28 = conv2d_41 + conv2d_40;  conv2d_41 = conv2d_40 = None
        truediv_11 = add_28 / 1.0;  add_28 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2203, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_3 = torch.cat([truediv_11, truediv_7], dim = 1);  truediv_11 = truediv_7 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_0_resnets_1_norm1 = self.self_up_blocks_0_resnets_1_norm1(cat_3)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_0_resnets_1_nonlinearity = self.self_up_blocks_0_resnets_1_nonlinearity(self_up_blocks_0_resnets_1_norm1);  self_up_blocks_0_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_1_conv1_weight = self.self_up_blocks_0_resnets_1_conv1_weight
        self_up_blocks_0_resnets_1_conv1_bias = self.self_up_blocks_0_resnets_1_conv1_bias
        conv2d_42 = torch.conv2d(self_up_blocks_0_resnets_1_nonlinearity, self_up_blocks_0_resnets_1_conv1_weight, self_up_blocks_0_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_0_resnets_1_nonlinearity = self_up_blocks_0_resnets_1_conv1_weight = self_up_blocks_0_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_0_resnets_1_nonlinearity_1 = self.self_up_blocks_0_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_0_resnets_1_time_emb_proj_weight = self.self_up_blocks_0_resnets_1_time_emb_proj_weight
        self_up_blocks_0_resnets_1_time_emb_proj_bias = self.self_up_blocks_0_resnets_1_time_emb_proj_bias
        linear_11 = torch._C._nn.linear(self_up_blocks_0_resnets_1_nonlinearity_1, self_up_blocks_0_resnets_1_time_emb_proj_weight, self_up_blocks_0_resnets_1_time_emb_proj_bias);  self_up_blocks_0_resnets_1_nonlinearity_1 = self_up_blocks_0_resnets_1_time_emb_proj_weight = self_up_blocks_0_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_16 = linear_11[(slice(None, None, None), slice(None, None, None), None, None)];  linear_11 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_29 = conv2d_42 + getitem_16;  conv2d_42 = getitem_16 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_0_resnets_1_norm2 = self.self_up_blocks_0_resnets_1_norm2(add_29);  add_29 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_0_resnets_1_nonlinearity_2 = self.self_up_blocks_0_resnets_1_nonlinearity(self_up_blocks_0_resnets_1_norm2);  self_up_blocks_0_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_0_resnets_1_dropout = self.self_up_blocks_0_resnets_1_dropout(self_up_blocks_0_resnets_1_nonlinearity_2);  self_up_blocks_0_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_1_conv2_weight = self.self_up_blocks_0_resnets_1_conv2_weight
        self_up_blocks_0_resnets_1_conv2_bias = self.self_up_blocks_0_resnets_1_conv2_bias
        conv2d_43 = torch.conv2d(self_up_blocks_0_resnets_1_dropout, self_up_blocks_0_resnets_1_conv2_weight, self_up_blocks_0_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_0_resnets_1_dropout = self_up_blocks_0_resnets_1_conv2_weight = self_up_blocks_0_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_1_conv_shortcut_weight = self.self_up_blocks_0_resnets_1_conv_shortcut_weight
        self_up_blocks_0_resnets_1_conv_shortcut_bias = self.self_up_blocks_0_resnets_1_conv_shortcut_bias
        conv2d_44 = torch.conv2d(cat_3, self_up_blocks_0_resnets_1_conv_shortcut_weight, self_up_blocks_0_resnets_1_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_3 = self_up_blocks_0_resnets_1_conv_shortcut_weight = self_up_blocks_0_resnets_1_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_30 = conv2d_44 + conv2d_43;  conv2d_44 = conv2d_43 = None
        truediv_12 = add_30 / 1.0;  add_30 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2203, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_4 = torch.cat([truediv_12, conv2d_28], dim = 1);  truediv_12 = conv2d_28 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_0_resnets_2_norm1 = self.self_up_blocks_0_resnets_2_norm1(cat_4)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_0_resnets_2_nonlinearity = self.self_up_blocks_0_resnets_2_nonlinearity(self_up_blocks_0_resnets_2_norm1);  self_up_blocks_0_resnets_2_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_2_conv1_weight = self.self_up_blocks_0_resnets_2_conv1_weight
        self_up_blocks_0_resnets_2_conv1_bias = self.self_up_blocks_0_resnets_2_conv1_bias
        conv2d_45 = torch.conv2d(self_up_blocks_0_resnets_2_nonlinearity, self_up_blocks_0_resnets_2_conv1_weight, self_up_blocks_0_resnets_2_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_0_resnets_2_nonlinearity = self_up_blocks_0_resnets_2_conv1_weight = self_up_blocks_0_resnets_2_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_0_resnets_2_nonlinearity_1 = self.self_up_blocks_0_resnets_2_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_0_resnets_2_time_emb_proj_weight = self.self_up_blocks_0_resnets_2_time_emb_proj_weight
        self_up_blocks_0_resnets_2_time_emb_proj_bias = self.self_up_blocks_0_resnets_2_time_emb_proj_bias
        linear_12 = torch._C._nn.linear(self_up_blocks_0_resnets_2_nonlinearity_1, self_up_blocks_0_resnets_2_time_emb_proj_weight, self_up_blocks_0_resnets_2_time_emb_proj_bias);  self_up_blocks_0_resnets_2_nonlinearity_1 = self_up_blocks_0_resnets_2_time_emb_proj_weight = self_up_blocks_0_resnets_2_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_17 = linear_12[(slice(None, None, None), slice(None, None, None), None, None)];  linear_12 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_31 = conv2d_45 + getitem_17;  conv2d_45 = getitem_17 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_0_resnets_2_norm2 = self.self_up_blocks_0_resnets_2_norm2(add_31);  add_31 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_0_resnets_2_nonlinearity_2 = self.self_up_blocks_0_resnets_2_nonlinearity(self_up_blocks_0_resnets_2_norm2);  self_up_blocks_0_resnets_2_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_0_resnets_2_dropout = self.self_up_blocks_0_resnets_2_dropout(self_up_blocks_0_resnets_2_nonlinearity_2);  self_up_blocks_0_resnets_2_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_2_conv2_weight = self.self_up_blocks_0_resnets_2_conv2_weight
        self_up_blocks_0_resnets_2_conv2_bias = self.self_up_blocks_0_resnets_2_conv2_bias
        conv2d_46 = torch.conv2d(self_up_blocks_0_resnets_2_dropout, self_up_blocks_0_resnets_2_conv2_weight, self_up_blocks_0_resnets_2_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_0_resnets_2_dropout = self_up_blocks_0_resnets_2_conv2_weight = self_up_blocks_0_resnets_2_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_resnets_2_conv_shortcut_weight = self.self_up_blocks_0_resnets_2_conv_shortcut_weight
        self_up_blocks_0_resnets_2_conv_shortcut_bias = self.self_up_blocks_0_resnets_2_conv_shortcut_bias
        conv2d_47 = torch.conv2d(cat_4, self_up_blocks_0_resnets_2_conv_shortcut_weight, self_up_blocks_0_resnets_2_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_4 = self_up_blocks_0_resnets_2_conv_shortcut_weight = self_up_blocks_0_resnets_2_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_32 = conv2d_47 + conv2d_46;  conv2d_47 = conv2d_46 = None
        truediv_13 = add_32 / 1.0;  add_32 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:158, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        interpolate = torch.nn.functional.interpolate(truediv_13, scale_factor = 2.0, mode = 'nearest');  truediv_13 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_0_upsamplers_0_conv_weight = self.self_up_blocks_0_upsamplers_0_conv_weight
        self_up_blocks_0_upsamplers_0_conv_bias = self.self_up_blocks_0_upsamplers_0_conv_bias
        conv2d_48 = torch.conv2d(interpolate, self_up_blocks_0_upsamplers_0_conv_weight, self_up_blocks_0_upsamplers_0_conv_bias, (1, 1), (1, 1), (1, 1), 1);  interpolate = self_up_blocks_0_upsamplers_0_conv_weight = self_up_blocks_0_upsamplers_0_conv_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_5 = torch.cat([conv2d_48, add_17], dim = 1);  conv2d_48 = add_17 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_1_resnets_0_norm1 = self.self_up_blocks_1_resnets_0_norm1(cat_5)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_1_resnets_0_nonlinearity = self.self_up_blocks_1_resnets_0_nonlinearity(self_up_blocks_1_resnets_0_norm1);  self_up_blocks_1_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_0_conv1_weight = self.self_up_blocks_1_resnets_0_conv1_weight
        self_up_blocks_1_resnets_0_conv1_bias = self.self_up_blocks_1_resnets_0_conv1_bias
        conv2d_49 = torch.conv2d(self_up_blocks_1_resnets_0_nonlinearity, self_up_blocks_1_resnets_0_conv1_weight, self_up_blocks_1_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_1_resnets_0_nonlinearity = self_up_blocks_1_resnets_0_conv1_weight = self_up_blocks_1_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_1_resnets_0_nonlinearity_1 = self.self_up_blocks_1_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_1_resnets_0_time_emb_proj_weight = self.self_up_blocks_1_resnets_0_time_emb_proj_weight
        self_up_blocks_1_resnets_0_time_emb_proj_bias = self.self_up_blocks_1_resnets_0_time_emb_proj_bias
        linear_13 = torch._C._nn.linear(self_up_blocks_1_resnets_0_nonlinearity_1, self_up_blocks_1_resnets_0_time_emb_proj_weight, self_up_blocks_1_resnets_0_time_emb_proj_bias);  self_up_blocks_1_resnets_0_nonlinearity_1 = self_up_blocks_1_resnets_0_time_emb_proj_weight = self_up_blocks_1_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_18 = linear_13[(slice(None, None, None), slice(None, None, None), None, None)];  linear_13 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_33 = conv2d_49 + getitem_18;  conv2d_49 = getitem_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_1_resnets_0_norm2 = self.self_up_blocks_1_resnets_0_norm2(add_33);  add_33 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_1_resnets_0_nonlinearity_2 = self.self_up_blocks_1_resnets_0_nonlinearity(self_up_blocks_1_resnets_0_norm2);  self_up_blocks_1_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_1_resnets_0_dropout = self.self_up_blocks_1_resnets_0_dropout(self_up_blocks_1_resnets_0_nonlinearity_2);  self_up_blocks_1_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_0_conv2_weight = self.self_up_blocks_1_resnets_0_conv2_weight
        self_up_blocks_1_resnets_0_conv2_bias = self.self_up_blocks_1_resnets_0_conv2_bias
        conv2d_50 = torch.conv2d(self_up_blocks_1_resnets_0_dropout, self_up_blocks_1_resnets_0_conv2_weight, self_up_blocks_1_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_1_resnets_0_dropout = self_up_blocks_1_resnets_0_conv2_weight = self_up_blocks_1_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_0_conv_shortcut_weight = self.self_up_blocks_1_resnets_0_conv_shortcut_weight
        self_up_blocks_1_resnets_0_conv_shortcut_bias = self.self_up_blocks_1_resnets_0_conv_shortcut_bias
        conv2d_51 = torch.conv2d(cat_5, self_up_blocks_1_resnets_0_conv_shortcut_weight, self_up_blocks_1_resnets_0_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_5 = self_up_blocks_1_resnets_0_conv_shortcut_weight = self_up_blocks_1_resnets_0_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_34 = conv2d_51 + conv2d_50;  conv2d_51 = conv2d_50 = None
        truediv_14 = add_34 / 1.0;  add_34 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_1_attentions_0_norm = self.self_up_blocks_1_attentions_0_norm(truediv_14)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_attentions_0_proj_in_weight = self.self_up_blocks_1_attentions_0_proj_in_weight
        self_up_blocks_1_attentions_0_proj_in_bias = self.self_up_blocks_1_attentions_0_proj_in_bias
        conv2d_52 = torch.conv2d(self_up_blocks_1_attentions_0_norm, self_up_blocks_1_attentions_0_proj_in_weight, self_up_blocks_1_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_1_attentions_0_norm = self_up_blocks_1_attentions_0_proj_in_weight = self_up_blocks_1_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_14 = conv2d_52.permute(0, 2, 3, 1);  conv2d_52 = None
        reshape_14 = permute_14.reshape(2, 256, 1280);  permute_14 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_1_attentions_0_transformer_blocks_0 = self.self_up_blocks_1_attentions_0_transformer_blocks_0(reshape_14, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_14 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_15 = self_up_blocks_1_attentions_0_transformer_blocks_0.reshape(2, 16, 16, 1280);  self_up_blocks_1_attentions_0_transformer_blocks_0 = None
        permute_15 = reshape_15.permute(0, 3, 1, 2);  reshape_15 = None
        contiguous_7 = permute_15.contiguous();  permute_15 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_attentions_0_proj_out_weight = self.self_up_blocks_1_attentions_0_proj_out_weight
        self_up_blocks_1_attentions_0_proj_out_bias = self.self_up_blocks_1_attentions_0_proj_out_bias
        conv2d_53 = torch.conv2d(contiguous_7, self_up_blocks_1_attentions_0_proj_out_weight, self_up_blocks_1_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_7 = self_up_blocks_1_attentions_0_proj_out_weight = self_up_blocks_1_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_35 = conv2d_53 + truediv_14;  conv2d_53 = truediv_14 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_6 = torch.cat([add_35, add_14], dim = 1);  add_35 = add_14 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_1_resnets_1_norm1 = self.self_up_blocks_1_resnets_1_norm1(cat_6)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_1_resnets_1_nonlinearity = self.self_up_blocks_1_resnets_1_nonlinearity(self_up_blocks_1_resnets_1_norm1);  self_up_blocks_1_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_1_conv1_weight = self.self_up_blocks_1_resnets_1_conv1_weight
        self_up_blocks_1_resnets_1_conv1_bias = self.self_up_blocks_1_resnets_1_conv1_bias
        conv2d_54 = torch.conv2d(self_up_blocks_1_resnets_1_nonlinearity, self_up_blocks_1_resnets_1_conv1_weight, self_up_blocks_1_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_1_resnets_1_nonlinearity = self_up_blocks_1_resnets_1_conv1_weight = self_up_blocks_1_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_1_resnets_1_nonlinearity_1 = self.self_up_blocks_1_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_1_resnets_1_time_emb_proj_weight = self.self_up_blocks_1_resnets_1_time_emb_proj_weight
        self_up_blocks_1_resnets_1_time_emb_proj_bias = self.self_up_blocks_1_resnets_1_time_emb_proj_bias
        linear_14 = torch._C._nn.linear(self_up_blocks_1_resnets_1_nonlinearity_1, self_up_blocks_1_resnets_1_time_emb_proj_weight, self_up_blocks_1_resnets_1_time_emb_proj_bias);  self_up_blocks_1_resnets_1_nonlinearity_1 = self_up_blocks_1_resnets_1_time_emb_proj_weight = self_up_blocks_1_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_19 = linear_14[(slice(None, None, None), slice(None, None, None), None, None)];  linear_14 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_36 = conv2d_54 + getitem_19;  conv2d_54 = getitem_19 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_1_resnets_1_norm2 = self.self_up_blocks_1_resnets_1_norm2(add_36);  add_36 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_1_resnets_1_nonlinearity_2 = self.self_up_blocks_1_resnets_1_nonlinearity(self_up_blocks_1_resnets_1_norm2);  self_up_blocks_1_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_1_resnets_1_dropout = self.self_up_blocks_1_resnets_1_dropout(self_up_blocks_1_resnets_1_nonlinearity_2);  self_up_blocks_1_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_1_conv2_weight = self.self_up_blocks_1_resnets_1_conv2_weight
        self_up_blocks_1_resnets_1_conv2_bias = self.self_up_blocks_1_resnets_1_conv2_bias
        conv2d_55 = torch.conv2d(self_up_blocks_1_resnets_1_dropout, self_up_blocks_1_resnets_1_conv2_weight, self_up_blocks_1_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_1_resnets_1_dropout = self_up_blocks_1_resnets_1_conv2_weight = self_up_blocks_1_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_1_conv_shortcut_weight = self.self_up_blocks_1_resnets_1_conv_shortcut_weight
        self_up_blocks_1_resnets_1_conv_shortcut_bias = self.self_up_blocks_1_resnets_1_conv_shortcut_bias
        conv2d_56 = torch.conv2d(cat_6, self_up_blocks_1_resnets_1_conv_shortcut_weight, self_up_blocks_1_resnets_1_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_6 = self_up_blocks_1_resnets_1_conv_shortcut_weight = self_up_blocks_1_resnets_1_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_37 = conv2d_56 + conv2d_55;  conv2d_56 = conv2d_55 = None
        truediv_15 = add_37 / 1.0;  add_37 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_1_attentions_1_norm = self.self_up_blocks_1_attentions_1_norm(truediv_15)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_attentions_1_proj_in_weight = self.self_up_blocks_1_attentions_1_proj_in_weight
        self_up_blocks_1_attentions_1_proj_in_bias = self.self_up_blocks_1_attentions_1_proj_in_bias
        conv2d_57 = torch.conv2d(self_up_blocks_1_attentions_1_norm, self_up_blocks_1_attentions_1_proj_in_weight, self_up_blocks_1_attentions_1_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_1_attentions_1_norm = self_up_blocks_1_attentions_1_proj_in_weight = self_up_blocks_1_attentions_1_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_16 = conv2d_57.permute(0, 2, 3, 1);  conv2d_57 = None
        reshape_16 = permute_16.reshape(2, 256, 1280);  permute_16 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_1_attentions_1_transformer_blocks_0 = self.self_up_blocks_1_attentions_1_transformer_blocks_0(reshape_16, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_16 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_17 = self_up_blocks_1_attentions_1_transformer_blocks_0.reshape(2, 16, 16, 1280);  self_up_blocks_1_attentions_1_transformer_blocks_0 = None
        permute_17 = reshape_17.permute(0, 3, 1, 2);  reshape_17 = None
        contiguous_8 = permute_17.contiguous();  permute_17 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_attentions_1_proj_out_weight = self.self_up_blocks_1_attentions_1_proj_out_weight
        self_up_blocks_1_attentions_1_proj_out_bias = self.self_up_blocks_1_attentions_1_proj_out_bias
        conv2d_58 = torch.conv2d(contiguous_8, self_up_blocks_1_attentions_1_proj_out_weight, self_up_blocks_1_attentions_1_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_8 = self_up_blocks_1_attentions_1_proj_out_weight = self_up_blocks_1_attentions_1_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_38 = conv2d_58 + truediv_15;  conv2d_58 = truediv_15 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_7 = torch.cat([add_38, conv2d_18], dim = 1);  add_38 = conv2d_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_1_resnets_2_norm1 = self.self_up_blocks_1_resnets_2_norm1(cat_7)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_1_resnets_2_nonlinearity = self.self_up_blocks_1_resnets_2_nonlinearity(self_up_blocks_1_resnets_2_norm1);  self_up_blocks_1_resnets_2_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_2_conv1_weight = self.self_up_blocks_1_resnets_2_conv1_weight
        self_up_blocks_1_resnets_2_conv1_bias = self.self_up_blocks_1_resnets_2_conv1_bias
        conv2d_59 = torch.conv2d(self_up_blocks_1_resnets_2_nonlinearity, self_up_blocks_1_resnets_2_conv1_weight, self_up_blocks_1_resnets_2_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_1_resnets_2_nonlinearity = self_up_blocks_1_resnets_2_conv1_weight = self_up_blocks_1_resnets_2_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_1_resnets_2_nonlinearity_1 = self.self_up_blocks_1_resnets_2_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_1_resnets_2_time_emb_proj_weight = self.self_up_blocks_1_resnets_2_time_emb_proj_weight
        self_up_blocks_1_resnets_2_time_emb_proj_bias = self.self_up_blocks_1_resnets_2_time_emb_proj_bias
        linear_15 = torch._C._nn.linear(self_up_blocks_1_resnets_2_nonlinearity_1, self_up_blocks_1_resnets_2_time_emb_proj_weight, self_up_blocks_1_resnets_2_time_emb_proj_bias);  self_up_blocks_1_resnets_2_nonlinearity_1 = self_up_blocks_1_resnets_2_time_emb_proj_weight = self_up_blocks_1_resnets_2_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_20 = linear_15[(slice(None, None, None), slice(None, None, None), None, None)];  linear_15 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_39 = conv2d_59 + getitem_20;  conv2d_59 = getitem_20 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_1_resnets_2_norm2 = self.self_up_blocks_1_resnets_2_norm2(add_39);  add_39 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_1_resnets_2_nonlinearity_2 = self.self_up_blocks_1_resnets_2_nonlinearity(self_up_blocks_1_resnets_2_norm2);  self_up_blocks_1_resnets_2_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_1_resnets_2_dropout = self.self_up_blocks_1_resnets_2_dropout(self_up_blocks_1_resnets_2_nonlinearity_2);  self_up_blocks_1_resnets_2_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_2_conv2_weight = self.self_up_blocks_1_resnets_2_conv2_weight
        self_up_blocks_1_resnets_2_conv2_bias = self.self_up_blocks_1_resnets_2_conv2_bias
        conv2d_60 = torch.conv2d(self_up_blocks_1_resnets_2_dropout, self_up_blocks_1_resnets_2_conv2_weight, self_up_blocks_1_resnets_2_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_1_resnets_2_dropout = self_up_blocks_1_resnets_2_conv2_weight = self_up_blocks_1_resnets_2_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_resnets_2_conv_shortcut_weight = self.self_up_blocks_1_resnets_2_conv_shortcut_weight
        self_up_blocks_1_resnets_2_conv_shortcut_bias = self.self_up_blocks_1_resnets_2_conv_shortcut_bias
        conv2d_61 = torch.conv2d(cat_7, self_up_blocks_1_resnets_2_conv_shortcut_weight, self_up_blocks_1_resnets_2_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_7 = self_up_blocks_1_resnets_2_conv_shortcut_weight = self_up_blocks_1_resnets_2_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_40 = conv2d_61 + conv2d_60;  conv2d_61 = conv2d_60 = None
        truediv_16 = add_40 / 1.0;  add_40 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_1_attentions_2_norm = self.self_up_blocks_1_attentions_2_norm(truediv_16)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_attentions_2_proj_in_weight = self.self_up_blocks_1_attentions_2_proj_in_weight
        self_up_blocks_1_attentions_2_proj_in_bias = self.self_up_blocks_1_attentions_2_proj_in_bias
        conv2d_62 = torch.conv2d(self_up_blocks_1_attentions_2_norm, self_up_blocks_1_attentions_2_proj_in_weight, self_up_blocks_1_attentions_2_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_1_attentions_2_norm = self_up_blocks_1_attentions_2_proj_in_weight = self_up_blocks_1_attentions_2_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_18 = conv2d_62.permute(0, 2, 3, 1);  conv2d_62 = None
        reshape_18 = permute_18.reshape(2, 256, 1280);  permute_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_1_attentions_2_transformer_blocks_0 = self.self_up_blocks_1_attentions_2_transformer_blocks_0(reshape_18, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_19 = self_up_blocks_1_attentions_2_transformer_blocks_0.reshape(2, 16, 16, 1280);  self_up_blocks_1_attentions_2_transformer_blocks_0 = None
        permute_19 = reshape_19.permute(0, 3, 1, 2);  reshape_19 = None
        contiguous_9 = permute_19.contiguous();  permute_19 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_attentions_2_proj_out_weight = self.self_up_blocks_1_attentions_2_proj_out_weight
        self_up_blocks_1_attentions_2_proj_out_bias = self.self_up_blocks_1_attentions_2_proj_out_bias
        conv2d_63 = torch.conv2d(contiguous_9, self_up_blocks_1_attentions_2_proj_out_weight, self_up_blocks_1_attentions_2_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_9 = self_up_blocks_1_attentions_2_proj_out_weight = self_up_blocks_1_attentions_2_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_41 = conv2d_63 + truediv_16;  conv2d_63 = truediv_16 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:158, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        interpolate_1 = torch.nn.functional.interpolate(add_41, scale_factor = 2.0, mode = 'nearest');  add_41 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_1_upsamplers_0_conv_weight = self.self_up_blocks_1_upsamplers_0_conv_weight
        self_up_blocks_1_upsamplers_0_conv_bias = self.self_up_blocks_1_upsamplers_0_conv_bias
        conv2d_64 = torch.conv2d(interpolate_1, self_up_blocks_1_upsamplers_0_conv_weight, self_up_blocks_1_upsamplers_0_conv_bias, (1, 1), (1, 1), (1, 1), 1);  interpolate_1 = self_up_blocks_1_upsamplers_0_conv_weight = self_up_blocks_1_upsamplers_0_conv_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_8 = torch.cat([conv2d_64, add_11], dim = 1);  conv2d_64 = add_11 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_2_resnets_0_norm1 = self.self_up_blocks_2_resnets_0_norm1(cat_8)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_2_resnets_0_nonlinearity = self.self_up_blocks_2_resnets_0_nonlinearity(self_up_blocks_2_resnets_0_norm1);  self_up_blocks_2_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_0_conv1_weight = self.self_up_blocks_2_resnets_0_conv1_weight
        self_up_blocks_2_resnets_0_conv1_bias = self.self_up_blocks_2_resnets_0_conv1_bias
        conv2d_65 = torch.conv2d(self_up_blocks_2_resnets_0_nonlinearity, self_up_blocks_2_resnets_0_conv1_weight, self_up_blocks_2_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_2_resnets_0_nonlinearity = self_up_blocks_2_resnets_0_conv1_weight = self_up_blocks_2_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_2_resnets_0_nonlinearity_1 = self.self_up_blocks_2_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_2_resnets_0_time_emb_proj_weight = self.self_up_blocks_2_resnets_0_time_emb_proj_weight
        self_up_blocks_2_resnets_0_time_emb_proj_bias = self.self_up_blocks_2_resnets_0_time_emb_proj_bias
        linear_16 = torch._C._nn.linear(self_up_blocks_2_resnets_0_nonlinearity_1, self_up_blocks_2_resnets_0_time_emb_proj_weight, self_up_blocks_2_resnets_0_time_emb_proj_bias);  self_up_blocks_2_resnets_0_nonlinearity_1 = self_up_blocks_2_resnets_0_time_emb_proj_weight = self_up_blocks_2_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_21 = linear_16[(slice(None, None, None), slice(None, None, None), None, None)];  linear_16 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_42 = conv2d_65 + getitem_21;  conv2d_65 = getitem_21 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_2_resnets_0_norm2 = self.self_up_blocks_2_resnets_0_norm2(add_42);  add_42 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_2_resnets_0_nonlinearity_2 = self.self_up_blocks_2_resnets_0_nonlinearity(self_up_blocks_2_resnets_0_norm2);  self_up_blocks_2_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_2_resnets_0_dropout = self.self_up_blocks_2_resnets_0_dropout(self_up_blocks_2_resnets_0_nonlinearity_2);  self_up_blocks_2_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_0_conv2_weight = self.self_up_blocks_2_resnets_0_conv2_weight
        self_up_blocks_2_resnets_0_conv2_bias = self.self_up_blocks_2_resnets_0_conv2_bias
        conv2d_66 = torch.conv2d(self_up_blocks_2_resnets_0_dropout, self_up_blocks_2_resnets_0_conv2_weight, self_up_blocks_2_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_2_resnets_0_dropout = self_up_blocks_2_resnets_0_conv2_weight = self_up_blocks_2_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_0_conv_shortcut_weight = self.self_up_blocks_2_resnets_0_conv_shortcut_weight
        self_up_blocks_2_resnets_0_conv_shortcut_bias = self.self_up_blocks_2_resnets_0_conv_shortcut_bias
        conv2d_67 = torch.conv2d(cat_8, self_up_blocks_2_resnets_0_conv_shortcut_weight, self_up_blocks_2_resnets_0_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_8 = self_up_blocks_2_resnets_0_conv_shortcut_weight = self_up_blocks_2_resnets_0_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_43 = conv2d_67 + conv2d_66;  conv2d_67 = conv2d_66 = None
        truediv_17 = add_43 / 1.0;  add_43 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_2_attentions_0_norm = self.self_up_blocks_2_attentions_0_norm(truediv_17)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_attentions_0_proj_in_weight = self.self_up_blocks_2_attentions_0_proj_in_weight
        self_up_blocks_2_attentions_0_proj_in_bias = self.self_up_blocks_2_attentions_0_proj_in_bias
        conv2d_68 = torch.conv2d(self_up_blocks_2_attentions_0_norm, self_up_blocks_2_attentions_0_proj_in_weight, self_up_blocks_2_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_2_attentions_0_norm = self_up_blocks_2_attentions_0_proj_in_weight = self_up_blocks_2_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_20 = conv2d_68.permute(0, 2, 3, 1);  conv2d_68 = None
        reshape_20 = permute_20.reshape(2, 1024, 640);  permute_20 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_2_attentions_0_transformer_blocks_0 = self.self_up_blocks_2_attentions_0_transformer_blocks_0(reshape_20, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_20 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_21 = self_up_blocks_2_attentions_0_transformer_blocks_0.reshape(2, 32, 32, 640);  self_up_blocks_2_attentions_0_transformer_blocks_0 = None
        permute_21 = reshape_21.permute(0, 3, 1, 2);  reshape_21 = None
        contiguous_10 = permute_21.contiguous();  permute_21 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_attentions_0_proj_out_weight = self.self_up_blocks_2_attentions_0_proj_out_weight
        self_up_blocks_2_attentions_0_proj_out_bias = self.self_up_blocks_2_attentions_0_proj_out_bias
        conv2d_69 = torch.conv2d(contiguous_10, self_up_blocks_2_attentions_0_proj_out_weight, self_up_blocks_2_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_10 = self_up_blocks_2_attentions_0_proj_out_weight = self_up_blocks_2_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_44 = conv2d_69 + truediv_17;  conv2d_69 = truediv_17 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_9 = torch.cat([add_44, add_8], dim = 1);  add_44 = add_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_2_resnets_1_norm1 = self.self_up_blocks_2_resnets_1_norm1(cat_9)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_2_resnets_1_nonlinearity = self.self_up_blocks_2_resnets_1_nonlinearity(self_up_blocks_2_resnets_1_norm1);  self_up_blocks_2_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_1_conv1_weight = self.self_up_blocks_2_resnets_1_conv1_weight
        self_up_blocks_2_resnets_1_conv1_bias = self.self_up_blocks_2_resnets_1_conv1_bias
        conv2d_70 = torch.conv2d(self_up_blocks_2_resnets_1_nonlinearity, self_up_blocks_2_resnets_1_conv1_weight, self_up_blocks_2_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_2_resnets_1_nonlinearity = self_up_blocks_2_resnets_1_conv1_weight = self_up_blocks_2_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_2_resnets_1_nonlinearity_1 = self.self_up_blocks_2_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_2_resnets_1_time_emb_proj_weight = self.self_up_blocks_2_resnets_1_time_emb_proj_weight
        self_up_blocks_2_resnets_1_time_emb_proj_bias = self.self_up_blocks_2_resnets_1_time_emb_proj_bias
        linear_17 = torch._C._nn.linear(self_up_blocks_2_resnets_1_nonlinearity_1, self_up_blocks_2_resnets_1_time_emb_proj_weight, self_up_blocks_2_resnets_1_time_emb_proj_bias);  self_up_blocks_2_resnets_1_nonlinearity_1 = self_up_blocks_2_resnets_1_time_emb_proj_weight = self_up_blocks_2_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_22 = linear_17[(slice(None, None, None), slice(None, None, None), None, None)];  linear_17 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_45 = conv2d_70 + getitem_22;  conv2d_70 = getitem_22 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_2_resnets_1_norm2 = self.self_up_blocks_2_resnets_1_norm2(add_45);  add_45 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_2_resnets_1_nonlinearity_2 = self.self_up_blocks_2_resnets_1_nonlinearity(self_up_blocks_2_resnets_1_norm2);  self_up_blocks_2_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_2_resnets_1_dropout = self.self_up_blocks_2_resnets_1_dropout(self_up_blocks_2_resnets_1_nonlinearity_2);  self_up_blocks_2_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_1_conv2_weight = self.self_up_blocks_2_resnets_1_conv2_weight
        self_up_blocks_2_resnets_1_conv2_bias = self.self_up_blocks_2_resnets_1_conv2_bias
        conv2d_71 = torch.conv2d(self_up_blocks_2_resnets_1_dropout, self_up_blocks_2_resnets_1_conv2_weight, self_up_blocks_2_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_2_resnets_1_dropout = self_up_blocks_2_resnets_1_conv2_weight = self_up_blocks_2_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_1_conv_shortcut_weight = self.self_up_blocks_2_resnets_1_conv_shortcut_weight
        self_up_blocks_2_resnets_1_conv_shortcut_bias = self.self_up_blocks_2_resnets_1_conv_shortcut_bias
        conv2d_72 = torch.conv2d(cat_9, self_up_blocks_2_resnets_1_conv_shortcut_weight, self_up_blocks_2_resnets_1_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_9 = self_up_blocks_2_resnets_1_conv_shortcut_weight = self_up_blocks_2_resnets_1_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_46 = conv2d_72 + conv2d_71;  conv2d_72 = conv2d_71 = None
        truediv_18 = add_46 / 1.0;  add_46 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_2_attentions_1_norm = self.self_up_blocks_2_attentions_1_norm(truediv_18)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_attentions_1_proj_in_weight = self.self_up_blocks_2_attentions_1_proj_in_weight
        self_up_blocks_2_attentions_1_proj_in_bias = self.self_up_blocks_2_attentions_1_proj_in_bias
        conv2d_73 = torch.conv2d(self_up_blocks_2_attentions_1_norm, self_up_blocks_2_attentions_1_proj_in_weight, self_up_blocks_2_attentions_1_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_2_attentions_1_norm = self_up_blocks_2_attentions_1_proj_in_weight = self_up_blocks_2_attentions_1_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_22 = conv2d_73.permute(0, 2, 3, 1);  conv2d_73 = None
        reshape_22 = permute_22.reshape(2, 1024, 640);  permute_22 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_2_attentions_1_transformer_blocks_0 = self.self_up_blocks_2_attentions_1_transformer_blocks_0(reshape_22, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_22 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_23 = self_up_blocks_2_attentions_1_transformer_blocks_0.reshape(2, 32, 32, 640);  self_up_blocks_2_attentions_1_transformer_blocks_0 = None
        permute_23 = reshape_23.permute(0, 3, 1, 2);  reshape_23 = None
        contiguous_11 = permute_23.contiguous();  permute_23 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_attentions_1_proj_out_weight = self.self_up_blocks_2_attentions_1_proj_out_weight
        self_up_blocks_2_attentions_1_proj_out_bias = self.self_up_blocks_2_attentions_1_proj_out_bias
        conv2d_74 = torch.conv2d(contiguous_11, self_up_blocks_2_attentions_1_proj_out_weight, self_up_blocks_2_attentions_1_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_11 = self_up_blocks_2_attentions_1_proj_out_weight = self_up_blocks_2_attentions_1_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_47 = conv2d_74 + truediv_18;  conv2d_74 = truediv_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_10 = torch.cat([add_47, conv2d_8], dim = 1);  add_47 = conv2d_8 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_2_resnets_2_norm1 = self.self_up_blocks_2_resnets_2_norm1(cat_10)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_2_resnets_2_nonlinearity = self.self_up_blocks_2_resnets_2_nonlinearity(self_up_blocks_2_resnets_2_norm1);  self_up_blocks_2_resnets_2_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_2_conv1_weight = self.self_up_blocks_2_resnets_2_conv1_weight
        self_up_blocks_2_resnets_2_conv1_bias = self.self_up_blocks_2_resnets_2_conv1_bias
        conv2d_75 = torch.conv2d(self_up_blocks_2_resnets_2_nonlinearity, self_up_blocks_2_resnets_2_conv1_weight, self_up_blocks_2_resnets_2_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_2_resnets_2_nonlinearity = self_up_blocks_2_resnets_2_conv1_weight = self_up_blocks_2_resnets_2_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_2_resnets_2_nonlinearity_1 = self.self_up_blocks_2_resnets_2_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_2_resnets_2_time_emb_proj_weight = self.self_up_blocks_2_resnets_2_time_emb_proj_weight
        self_up_blocks_2_resnets_2_time_emb_proj_bias = self.self_up_blocks_2_resnets_2_time_emb_proj_bias
        linear_18 = torch._C._nn.linear(self_up_blocks_2_resnets_2_nonlinearity_1, self_up_blocks_2_resnets_2_time_emb_proj_weight, self_up_blocks_2_resnets_2_time_emb_proj_bias);  self_up_blocks_2_resnets_2_nonlinearity_1 = self_up_blocks_2_resnets_2_time_emb_proj_weight = self_up_blocks_2_resnets_2_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_23 = linear_18[(slice(None, None, None), slice(None, None, None), None, None)];  linear_18 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_48 = conv2d_75 + getitem_23;  conv2d_75 = getitem_23 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_2_resnets_2_norm2 = self.self_up_blocks_2_resnets_2_norm2(add_48);  add_48 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_2_resnets_2_nonlinearity_2 = self.self_up_blocks_2_resnets_2_nonlinearity(self_up_blocks_2_resnets_2_norm2);  self_up_blocks_2_resnets_2_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_2_resnets_2_dropout = self.self_up_blocks_2_resnets_2_dropout(self_up_blocks_2_resnets_2_nonlinearity_2);  self_up_blocks_2_resnets_2_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_2_conv2_weight = self.self_up_blocks_2_resnets_2_conv2_weight
        self_up_blocks_2_resnets_2_conv2_bias = self.self_up_blocks_2_resnets_2_conv2_bias
        conv2d_76 = torch.conv2d(self_up_blocks_2_resnets_2_dropout, self_up_blocks_2_resnets_2_conv2_weight, self_up_blocks_2_resnets_2_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_2_resnets_2_dropout = self_up_blocks_2_resnets_2_conv2_weight = self_up_blocks_2_resnets_2_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_resnets_2_conv_shortcut_weight = self.self_up_blocks_2_resnets_2_conv_shortcut_weight
        self_up_blocks_2_resnets_2_conv_shortcut_bias = self.self_up_blocks_2_resnets_2_conv_shortcut_bias
        conv2d_77 = torch.conv2d(cat_10, self_up_blocks_2_resnets_2_conv_shortcut_weight, self_up_blocks_2_resnets_2_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_10 = self_up_blocks_2_resnets_2_conv_shortcut_weight = self_up_blocks_2_resnets_2_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_49 = conv2d_77 + conv2d_76;  conv2d_77 = conv2d_76 = None
        truediv_19 = add_49 / 1.0;  add_49 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_2_attentions_2_norm = self.self_up_blocks_2_attentions_2_norm(truediv_19)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_attentions_2_proj_in_weight = self.self_up_blocks_2_attentions_2_proj_in_weight
        self_up_blocks_2_attentions_2_proj_in_bias = self.self_up_blocks_2_attentions_2_proj_in_bias
        conv2d_78 = torch.conv2d(self_up_blocks_2_attentions_2_norm, self_up_blocks_2_attentions_2_proj_in_weight, self_up_blocks_2_attentions_2_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_2_attentions_2_norm = self_up_blocks_2_attentions_2_proj_in_weight = self_up_blocks_2_attentions_2_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_24 = conv2d_78.permute(0, 2, 3, 1);  conv2d_78 = None
        reshape_24 = permute_24.reshape(2, 1024, 640);  permute_24 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_2_attentions_2_transformer_blocks_0 = self.self_up_blocks_2_attentions_2_transformer_blocks_0(reshape_24, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_24 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_25 = self_up_blocks_2_attentions_2_transformer_blocks_0.reshape(2, 32, 32, 640);  self_up_blocks_2_attentions_2_transformer_blocks_0 = None
        permute_25 = reshape_25.permute(0, 3, 1, 2);  reshape_25 = None
        contiguous_12 = permute_25.contiguous();  permute_25 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_attentions_2_proj_out_weight = self.self_up_blocks_2_attentions_2_proj_out_weight
        self_up_blocks_2_attentions_2_proj_out_bias = self.self_up_blocks_2_attentions_2_proj_out_bias
        conv2d_79 = torch.conv2d(contiguous_12, self_up_blocks_2_attentions_2_proj_out_weight, self_up_blocks_2_attentions_2_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_12 = self_up_blocks_2_attentions_2_proj_out_weight = self_up_blocks_2_attentions_2_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_50 = conv2d_79 + truediv_19;  conv2d_79 = truediv_19 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:158, code: hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        interpolate_2 = torch.nn.functional.interpolate(add_50, scale_factor = 2.0, mode = 'nearest');  add_50 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_2_upsamplers_0_conv_weight = self.self_up_blocks_2_upsamplers_0_conv_weight
        self_up_blocks_2_upsamplers_0_conv_bias = self.self_up_blocks_2_upsamplers_0_conv_bias
        conv2d_80 = torch.conv2d(interpolate_2, self_up_blocks_2_upsamplers_0_conv_weight, self_up_blocks_2_upsamplers_0_conv_bias, (1, 1), (1, 1), (1, 1), 1);  interpolate_2 = self_up_blocks_2_upsamplers_0_conv_weight = self_up_blocks_2_upsamplers_0_conv_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_11 = torch.cat([conv2d_80, add_5], dim = 1);  conv2d_80 = add_5 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_3_resnets_0_norm1 = self.self_up_blocks_3_resnets_0_norm1(cat_11)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_3_resnets_0_nonlinearity = self.self_up_blocks_3_resnets_0_nonlinearity(self_up_blocks_3_resnets_0_norm1);  self_up_blocks_3_resnets_0_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_0_conv1_weight = self.self_up_blocks_3_resnets_0_conv1_weight
        self_up_blocks_3_resnets_0_conv1_bias = self.self_up_blocks_3_resnets_0_conv1_bias
        conv2d_81 = torch.conv2d(self_up_blocks_3_resnets_0_nonlinearity, self_up_blocks_3_resnets_0_conv1_weight, self_up_blocks_3_resnets_0_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_3_resnets_0_nonlinearity = self_up_blocks_3_resnets_0_conv1_weight = self_up_blocks_3_resnets_0_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_3_resnets_0_nonlinearity_1 = self.self_up_blocks_3_resnets_0_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_3_resnets_0_time_emb_proj_weight = self.self_up_blocks_3_resnets_0_time_emb_proj_weight
        self_up_blocks_3_resnets_0_time_emb_proj_bias = self.self_up_blocks_3_resnets_0_time_emb_proj_bias
        linear_19 = torch._C._nn.linear(self_up_blocks_3_resnets_0_nonlinearity_1, self_up_blocks_3_resnets_0_time_emb_proj_weight, self_up_blocks_3_resnets_0_time_emb_proj_bias);  self_up_blocks_3_resnets_0_nonlinearity_1 = self_up_blocks_3_resnets_0_time_emb_proj_weight = self_up_blocks_3_resnets_0_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_24 = linear_19[(slice(None, None, None), slice(None, None, None), None, None)];  linear_19 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_51 = conv2d_81 + getitem_24;  conv2d_81 = getitem_24 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_3_resnets_0_norm2 = self.self_up_blocks_3_resnets_0_norm2(add_51);  add_51 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_3_resnets_0_nonlinearity_2 = self.self_up_blocks_3_resnets_0_nonlinearity(self_up_blocks_3_resnets_0_norm2);  self_up_blocks_3_resnets_0_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_3_resnets_0_dropout = self.self_up_blocks_3_resnets_0_dropout(self_up_blocks_3_resnets_0_nonlinearity_2);  self_up_blocks_3_resnets_0_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_0_conv2_weight = self.self_up_blocks_3_resnets_0_conv2_weight
        self_up_blocks_3_resnets_0_conv2_bias = self.self_up_blocks_3_resnets_0_conv2_bias
        conv2d_82 = torch.conv2d(self_up_blocks_3_resnets_0_dropout, self_up_blocks_3_resnets_0_conv2_weight, self_up_blocks_3_resnets_0_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_3_resnets_0_dropout = self_up_blocks_3_resnets_0_conv2_weight = self_up_blocks_3_resnets_0_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_0_conv_shortcut_weight = self.self_up_blocks_3_resnets_0_conv_shortcut_weight
        self_up_blocks_3_resnets_0_conv_shortcut_bias = self.self_up_blocks_3_resnets_0_conv_shortcut_bias
        conv2d_83 = torch.conv2d(cat_11, self_up_blocks_3_resnets_0_conv_shortcut_weight, self_up_blocks_3_resnets_0_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_11 = self_up_blocks_3_resnets_0_conv_shortcut_weight = self_up_blocks_3_resnets_0_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_52 = conv2d_83 + conv2d_82;  conv2d_83 = conv2d_82 = None
        truediv_20 = add_52 / 1.0;  add_52 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_3_attentions_0_norm = self.self_up_blocks_3_attentions_0_norm(truediv_20)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_attentions_0_proj_in_weight = self.self_up_blocks_3_attentions_0_proj_in_weight
        self_up_blocks_3_attentions_0_proj_in_bias = self.self_up_blocks_3_attentions_0_proj_in_bias
        conv2d_84 = torch.conv2d(self_up_blocks_3_attentions_0_norm, self_up_blocks_3_attentions_0_proj_in_weight, self_up_blocks_3_attentions_0_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_3_attentions_0_norm = self_up_blocks_3_attentions_0_proj_in_weight = self_up_blocks_3_attentions_0_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_26 = conv2d_84.permute(0, 2, 3, 1);  conv2d_84 = None
        reshape_26 = permute_26.reshape(2, 4096, 320);  permute_26 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_3_attentions_0_transformer_blocks_0 = self.self_up_blocks_3_attentions_0_transformer_blocks_0(reshape_26, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_26 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_27 = self_up_blocks_3_attentions_0_transformer_blocks_0.reshape(2, 64, 64, 320);  self_up_blocks_3_attentions_0_transformer_blocks_0 = None
        permute_27 = reshape_27.permute(0, 3, 1, 2);  reshape_27 = None
        contiguous_13 = permute_27.contiguous();  permute_27 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_attentions_0_proj_out_weight = self.self_up_blocks_3_attentions_0_proj_out_weight
        self_up_blocks_3_attentions_0_proj_out_bias = self.self_up_blocks_3_attentions_0_proj_out_bias
        conv2d_85 = torch.conv2d(contiguous_13, self_up_blocks_3_attentions_0_proj_out_weight, self_up_blocks_3_attentions_0_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_13 = self_up_blocks_3_attentions_0_proj_out_weight = self_up_blocks_3_attentions_0_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_53 = conv2d_85 + truediv_20;  conv2d_85 = truediv_20 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_12 = torch.cat([add_53, add_2], dim = 1);  add_53 = add_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_3_resnets_1_norm1 = self.self_up_blocks_3_resnets_1_norm1(cat_12)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_3_resnets_1_nonlinearity = self.self_up_blocks_3_resnets_1_nonlinearity(self_up_blocks_3_resnets_1_norm1);  self_up_blocks_3_resnets_1_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_1_conv1_weight = self.self_up_blocks_3_resnets_1_conv1_weight
        self_up_blocks_3_resnets_1_conv1_bias = self.self_up_blocks_3_resnets_1_conv1_bias
        conv2d_86 = torch.conv2d(self_up_blocks_3_resnets_1_nonlinearity, self_up_blocks_3_resnets_1_conv1_weight, self_up_blocks_3_resnets_1_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_3_resnets_1_nonlinearity = self_up_blocks_3_resnets_1_conv1_weight = self_up_blocks_3_resnets_1_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_3_resnets_1_nonlinearity_1 = self.self_up_blocks_3_resnets_1_nonlinearity(self_time_embedding_linear_2)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_3_resnets_1_time_emb_proj_weight = self.self_up_blocks_3_resnets_1_time_emb_proj_weight
        self_up_blocks_3_resnets_1_time_emb_proj_bias = self.self_up_blocks_3_resnets_1_time_emb_proj_bias
        linear_20 = torch._C._nn.linear(self_up_blocks_3_resnets_1_nonlinearity_1, self_up_blocks_3_resnets_1_time_emb_proj_weight, self_up_blocks_3_resnets_1_time_emb_proj_bias);  self_up_blocks_3_resnets_1_nonlinearity_1 = self_up_blocks_3_resnets_1_time_emb_proj_weight = self_up_blocks_3_resnets_1_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_25 = linear_20[(slice(None, None, None), slice(None, None, None), None, None)];  linear_20 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_54 = conv2d_86 + getitem_25;  conv2d_86 = getitem_25 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_3_resnets_1_norm2 = self.self_up_blocks_3_resnets_1_norm2(add_54);  add_54 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_3_resnets_1_nonlinearity_2 = self.self_up_blocks_3_resnets_1_nonlinearity(self_up_blocks_3_resnets_1_norm2);  self_up_blocks_3_resnets_1_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_3_resnets_1_dropout = self.self_up_blocks_3_resnets_1_dropout(self_up_blocks_3_resnets_1_nonlinearity_2);  self_up_blocks_3_resnets_1_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_1_conv2_weight = self.self_up_blocks_3_resnets_1_conv2_weight
        self_up_blocks_3_resnets_1_conv2_bias = self.self_up_blocks_3_resnets_1_conv2_bias
        conv2d_87 = torch.conv2d(self_up_blocks_3_resnets_1_dropout, self_up_blocks_3_resnets_1_conv2_weight, self_up_blocks_3_resnets_1_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_3_resnets_1_dropout = self_up_blocks_3_resnets_1_conv2_weight = self_up_blocks_3_resnets_1_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_1_conv_shortcut_weight = self.self_up_blocks_3_resnets_1_conv_shortcut_weight
        self_up_blocks_3_resnets_1_conv_shortcut_bias = self.self_up_blocks_3_resnets_1_conv_shortcut_bias
        conv2d_88 = torch.conv2d(cat_12, self_up_blocks_3_resnets_1_conv_shortcut_weight, self_up_blocks_3_resnets_1_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_12 = self_up_blocks_3_resnets_1_conv_shortcut_weight = self_up_blocks_3_resnets_1_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_55 = conv2d_88 + conv2d_87;  conv2d_88 = conv2d_87 = None
        truediv_21 = add_55 / 1.0;  add_55 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_3_attentions_1_norm = self.self_up_blocks_3_attentions_1_norm(truediv_21)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_attentions_1_proj_in_weight = self.self_up_blocks_3_attentions_1_proj_in_weight
        self_up_blocks_3_attentions_1_proj_in_bias = self.self_up_blocks_3_attentions_1_proj_in_bias
        conv2d_89 = torch.conv2d(self_up_blocks_3_attentions_1_norm, self_up_blocks_3_attentions_1_proj_in_weight, self_up_blocks_3_attentions_1_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_3_attentions_1_norm = self_up_blocks_3_attentions_1_proj_in_weight = self_up_blocks_3_attentions_1_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_28 = conv2d_89.permute(0, 2, 3, 1);  conv2d_89 = None
        reshape_28 = permute_28.reshape(2, 4096, 320);  permute_28 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_3_attentions_1_transformer_blocks_0 = self.self_up_blocks_3_attentions_1_transformer_blocks_0(reshape_28, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_28 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_29 = self_up_blocks_3_attentions_1_transformer_blocks_0.reshape(2, 64, 64, 320);  self_up_blocks_3_attentions_1_transformer_blocks_0 = None
        permute_29 = reshape_29.permute(0, 3, 1, 2);  reshape_29 = None
        contiguous_14 = permute_29.contiguous();  permute_29 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_attentions_1_proj_out_weight = self.self_up_blocks_3_attentions_1_proj_out_weight
        self_up_blocks_3_attentions_1_proj_out_bias = self.self_up_blocks_3_attentions_1_proj_out_bias
        conv2d_90 = torch.conv2d(contiguous_14, self_up_blocks_3_attentions_1_proj_out_weight, self_up_blocks_3_attentions_1_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_14 = self_up_blocks_3_attentions_1_proj_out_weight = self_up_blocks_3_attentions_1_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_56 = conv2d_90 + truediv_21;  conv2d_90 = truediv_21 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_blocks.py:2101, code: hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        cat_13 = torch.cat([add_56, self_conv_in], dim = 1);  add_56 = self_conv_in = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:597, code: hidden_states = self.norm1(hidden_states)
        self_up_blocks_3_resnets_2_norm1 = self.self_up_blocks_3_resnets_2_norm1(cat_13)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:599, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_3_resnets_2_nonlinearity = self.self_up_blocks_3_resnets_2_nonlinearity(self_up_blocks_3_resnets_2_norm1);  self_up_blocks_3_resnets_2_norm1 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_2_conv1_weight = self.self_up_blocks_3_resnets_2_conv1_weight
        self_up_blocks_3_resnets_2_conv1_bias = self.self_up_blocks_3_resnets_2_conv1_bias
        conv2d_91 = torch.conv2d(self_up_blocks_3_resnets_2_nonlinearity, self_up_blocks_3_resnets_2_conv1_weight, self_up_blocks_3_resnets_2_conv1_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_3_resnets_2_nonlinearity = self_up_blocks_3_resnets_2_conv1_weight = self_up_blocks_3_resnets_2_conv1_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:616, code: temb = self.nonlinearity(temb)
        self_up_blocks_3_resnets_2_nonlinearity_1 = self.self_up_blocks_3_resnets_2_nonlinearity(self_time_embedding_linear_2);  self_time_embedding_linear_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
        self_up_blocks_3_resnets_2_time_emb_proj_weight = self.self_up_blocks_3_resnets_2_time_emb_proj_weight
        self_up_blocks_3_resnets_2_time_emb_proj_bias = self.self_up_blocks_3_resnets_2_time_emb_proj_bias
        linear_21 = torch._C._nn.linear(self_up_blocks_3_resnets_2_nonlinearity_1, self_up_blocks_3_resnets_2_time_emb_proj_weight, self_up_blocks_3_resnets_2_time_emb_proj_bias);  self_up_blocks_3_resnets_2_nonlinearity_1 = self_up_blocks_3_resnets_2_time_emb_proj_weight = self_up_blocks_3_resnets_2_time_emb_proj_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:617, code: temb = self.time_emb_proj(temb)[:, :, None, None]
        getitem_26 = linear_21[(slice(None, None, None), slice(None, None, None), None, None)];  linear_21 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:620, code: hidden_states = hidden_states + temb
        add_57 = conv2d_91 + getitem_26;  conv2d_91 = getitem_26 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:625, code: hidden_states = self.norm2(hidden_states)
        self_up_blocks_3_resnets_2_norm2 = self.self_up_blocks_3_resnets_2_norm2(add_57);  add_57 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:631, code: hidden_states = self.nonlinearity(hidden_states)
        self_up_blocks_3_resnets_2_nonlinearity_2 = self.self_up_blocks_3_resnets_2_nonlinearity(self_up_blocks_3_resnets_2_norm2);  self_up_blocks_3_resnets_2_norm2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:633, code: hidden_states = self.dropout(hidden_states)
        self_up_blocks_3_resnets_2_dropout = self.self_up_blocks_3_resnets_2_dropout(self_up_blocks_3_resnets_2_nonlinearity_2);  self_up_blocks_3_resnets_2_nonlinearity_2 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_2_conv2_weight = self.self_up_blocks_3_resnets_2_conv2_weight
        self_up_blocks_3_resnets_2_conv2_bias = self.self_up_blocks_3_resnets_2_conv2_bias
        conv2d_92 = torch.conv2d(self_up_blocks_3_resnets_2_dropout, self_up_blocks_3_resnets_2_conv2_weight, self_up_blocks_3_resnets_2_conv2_bias, (1, 1), (1, 1), (1, 1), 1);  self_up_blocks_3_resnets_2_dropout = self_up_blocks_3_resnets_2_conv2_weight = self_up_blocks_3_resnets_2_conv2_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_resnets_2_conv_shortcut_weight = self.self_up_blocks_3_resnets_2_conv_shortcut_weight
        self_up_blocks_3_resnets_2_conv_shortcut_bias = self.self_up_blocks_3_resnets_2_conv_shortcut_bias
        conv2d_93 = torch.conv2d(cat_13, self_up_blocks_3_resnets_2_conv_shortcut_weight, self_up_blocks_3_resnets_2_conv_shortcut_bias, (1, 1), (0, 0), (1, 1), 1);  cat_13 = self_up_blocks_3_resnets_2_conv_shortcut_weight = self_up_blocks_3_resnets_2_conv_shortcut_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/resnet.py:639, code: output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        add_58 = conv2d_93 + conv2d_92;  conv2d_93 = conv2d_92 = None
        truediv_22 = add_58 / 1.0;  add_58 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:276, code: hidden_states = self.norm(hidden_states)
        self_up_blocks_3_attentions_2_norm = self.self_up_blocks_3_attentions_2_norm(truediv_22)

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_attentions_2_proj_in_weight = self.self_up_blocks_3_attentions_2_proj_in_weight
        self_up_blocks_3_attentions_2_proj_in_bias = self.self_up_blocks_3_attentions_2_proj_in_bias
        conv2d_94 = torch.conv2d(self_up_blocks_3_attentions_2_norm, self_up_blocks_3_attentions_2_proj_in_weight, self_up_blocks_3_attentions_2_proj_in_bias, (1, 1), (0, 0), (1, 1), 1);  self_up_blocks_3_attentions_2_norm = self_up_blocks_3_attentions_2_proj_in_weight = self_up_blocks_3_attentions_2_proj_in_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:280, code: hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        permute_30 = conv2d_94.permute(0, 2, 3, 1);  conv2d_94 = None
        reshape_30 = permute_30.reshape(2, 4096, 320);  permute_30 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:292, code: hidden_states = block(
        self_up_blocks_3_attentions_2_transformer_blocks_0 = self.self_up_blocks_3_attentions_2_transformer_blocks_0(reshape_30, attention_mask = None, encoder_hidden_states = encoder_hidden_states, encoder_attention_mask = None, timestep = None, cross_attention_kwargs = None, class_labels = None);  reshape_30 = encoder_hidden_states = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:305, code: hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
        reshape_31 = self_up_blocks_3_attentions_2_transformer_blocks_0.reshape(2, 64, 64, 320);  self_up_blocks_3_attentions_2_transformer_blocks_0 = None
        permute_31 = reshape_31.permute(0, 3, 1, 2);  reshape_31 = None
        contiguous_15 = permute_31.contiguous();  permute_31 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/lora.py:102, code: return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self_up_blocks_3_attentions_2_proj_out_weight = self.self_up_blocks_3_attentions_2_proj_out_weight
        self_up_blocks_3_attentions_2_proj_out_bias = self.self_up_blocks_3_attentions_2_proj_out_bias
        conv2d_95 = torch.conv2d(contiguous_15, self_up_blocks_3_attentions_2_proj_out_weight, self_up_blocks_3_attentions_2_proj_out_bias, (1, 1), (0, 0), (1, 1), 1);  contiguous_15 = self_up_blocks_3_attentions_2_proj_out_weight = self_up_blocks_3_attentions_2_proj_out_bias = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/transformer_2d.py:311, code: output = hidden_states + residual
        add_59 = conv2d_95 + truediv_22;  conv2d_95 = truediv_22 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:987, code: sample = self.conv_norm_out(sample)
        self_conv_norm_out = self.self_conv_norm_out(add_59);  add_59 = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:988, code: sample = self.conv_act(sample)
        self_conv_act = self.self_conv_act(self_conv_norm_out);  self_conv_norm_out = None

        # File: /home/caishenghang/miniconda3/envs/py310pt2/lib/python3.10/site-packages/diffusers/models/unet_2d_condition.py:989, code: sample = self.conv_out(sample)
        self_conv_out = self.self_conv_out(self_conv_act);  self_conv_act = None
        return (self_conv_out,)
