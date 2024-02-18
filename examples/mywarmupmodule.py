import oneflow as torch
import oneflow.nn as nn


class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()

        self.conv2d = nn.Conv2d(3, 6, 5)
        self.groupnorm = nn.GroupNorm(3, 6)
        self.layer_norm = nn.LayerNorm([24, 6, 24])

    def forward(self, x, a, b, w):
        x = self.conv2d(x)
        x = self.groupnorm(x)
        x = torch.transpose(x, 1, 2)
        x = self.layer_norm(x)
        x = torch.reshape(x, (2, -1))
        x = torch.sin(x)
        x = torch.cos(x)
        tup_list = [[None, None, None], [0, 4, 1],]
        x = torch.slice(x, slice_tup_list=tup_list)
        x = x.reshape(2, -1, 4)
        x = x + a
        x = torch.bmm(x, b)
        x = torch.softmax(x, dim=1)
        x = x*1.0
        x = x/1.0
        print(f"### x:{x.shape}, a:{a.shape}, b:{b.shape}, w:{w.shape}")
        x = torch.flatten(x, start_dim=1)
        a = torch.flatten(x, start_dim=1)
        print(f"### x:{x.shape}, a:{a.shape}, b:{b.shape}, w:{w.shape}")
        x = torch.cat([x,a])
        x = torch.clip(x, min=-1, max=1)
        x = torch.exp(x)
        x = torch.flatten(x, start_dim=0)
        tup_list = [[0, 32, 1],]
        x = torch.slice(x, slice_tup_list=tup_list)
        x = torch._C.unflatten(x, 0, (4, 8))
        a_1 = torch.empty(a.shape).cuda()
        a = a + a_1
        x = torch._C.reshape_like(x, a)
        x = torch.cast(x, torch.float16)
        print(f"### x:{x.shape} {x.dtype}, a:{a.shape}, b:{b.shape}, w:{w.shape}")
        x = torch.narrow(x, 0, 0, 2)
        return x
class SimpleGraph(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, *input):
        return self.model(*input)

def global_warmup():
    m = SimpleModule()
    m.eval()
    m = m.cuda()
    a = torch.randn(2, 4, 4).cuda()
    b = torch.randn(2, 4, 4).cuda()
    x = torch.randn(1,3,28,28).cuda()
    w = torch.randn(2,4).cuda()
    g = SimpleGraph(m)
    y = g(x,a,b,w)
    print(y.shape)

if __name__ == "__main__":
    global_warmup()