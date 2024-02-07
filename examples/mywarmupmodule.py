import oneflow as torch
import oneflow.nn as nn


class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()

        # Define parameters
        self.conv2d = nn.Conv2d(3, 6, 5)
        self.groupnorm = nn.GroupNorm(3, 6)
        self.layer_norm = nn.LayerNorm([24, 6, 24])

    def forward(self, x):
        # # Perform operations
        # #x = torch.expand_dims(x, dim=1)
        # #x = torch.broadcast_dim_like(x, self.weight)
        # x = torch.arange(10)
        # #x = torch.scalar_mul(x, 0.5)
        # #x = torch.scalar_div(x, 2.0)
        # x = torch.exp(x)
        # #x = torch.broadcast_mul(x, self.weight)
        # x = torch.sin(x)
        # x = torch.cos(x)
        # #x = torch.clip(x, -1, 0)
        # #x = torch.cat([x, x], dim=1)
        # #x = torch.slice(x, 1, 0, 5)
        # #x = torch.cast(x, dtype=torch.float16)
        # #x = torch.fused_matmul_bias(x, self.weight, self.bias)
        # #x = torch.silu(x)
        # x = torch.flatten(x)
        # #x = torch.unflatten(x, sizes=[2, 3, 4])
        # #x = torch.add_n([x, x])
        
        x = self.conv2d(x)
        x = self.groupnorm(x)
        #x = torch.broadcast_add(x, self.bias)
        x = torch.transpose(x, 1, 2)
        x = self.layer_norm(x)
        #x = torch.broadcast_matmul(x, x)
        x = torch.reshape(x, (2, -1))
        x = torch.empty(3, 3)
        
        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 8, 3)
        x = torch.bmm(a, b)
        x = torch.softmax(x, dim=1)
        x = torch.narrow(x, 1, 0, 2)
        #x = torch.gelu(x)
        #x = torch.reshape_like(x, self.weight)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        #x = torch.grouped_matmul_bias(x, self.weight, self.bias)
        #x = torch.timestep_embedding(x, 4)
        #x = torch.fused_multi_head_attention_inference(x, self.weight, self.bias)
        #x = torch.fused_glu(x)

        return x
class SimpleGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, input):
        return self.model(input)
        
if __name__ == "__main__":
    m = SimpleModule()
    m.eval()
    m = m.cuda()
    x = torch.randn(1,3,28,28).cuda()
    g = SimpleGraph(m)
    y = g(x)
    print(y.shape)