import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Conv2d(1, 192, 4, 2, 1, bias=False)
        self.tm_emb = nn.Linear(1, 192, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(197, 192))
        enc_layer = nn.TransformerEncoderLayer(192, 8, 288, 0.2, 'gelu', 1e-5, True)
        enc_layer.self_attn.out_proj.weight.detach().zero_()
        enc_layer.self_attn.out_proj.bias.detach().zero_()
        enc_layer.linear2.weight.detach().zero_()
        enc_layer.linear2.bias.detach().zero_()
        self.body = nn.TransformerEncoder(enc_layer, 32)
        self.tail = nn.Linear(192, 4, bias=False)
        self.tail.weight.detach().zero_()
    
    def forward(self, x, t):
        x = self.head(x).view(-1, 192, 196).mT
        t = self.tm_emb(t.reshape(-1, 1, 1))
        x = torch.cat((x, t), dim=1) + self.pos_emb
        x = self.tail(self.body(x)[:, :-1])
        x = x.view(-1, 14, 14, 2, 2).transpose(2, 3).reshape(-1, 1, 28, 28)
        return x


def train():
    cuda = torch.device('cuda:0')
    ds = torch.from_file('train-images-idx3-ubyte', False, 47040016, dtype=torch.uint8)
    ds = ds[16:].view(60000, 1, 28, 28).to(cuda)
    ds = torch.utils.data.TensorDataset(ds)
    dl = torch.utils.data.DataLoader(ds, batch_size=60, shuffle=True)
    
    model = DiT().to(cuda)
    opti = torch.optim.Adam(model.parameters())
    lr = torch.linspace(0, 1.5708, 101).cos().square().mul(5e-4).tolist()
    scaler = torch.cuda.amp.GradScaler()
    pbar = tqdm(total=100000)
    for epoch in range(100):
        for pg in opti.param_groups: pg['lr'] = lr[epoch]
        for x0, in dl:
            opti.zero_grad()
            x0 = (x0.float() - 33) / 79
            noise = torch.randn_like(x0)
            t = torch.rand(60, 1, 1, 1, device=cuda)
            x = (1 - t).sqrt() * x0 + t.sqrt() * noise
            with torch.cuda.amp.autocast():
                loss = nn.functional.mse_loss(model(x, t), noise)
            scaler.scale(loss).backward()
            scaler.step(opti)
            scaler.update()
            pbar.set_postfix(loss='%.4f'%loss.item())
            pbar.update(1)
        torch.save(model.state_dict(), 'f32.pt')
    torch.save(model.half().state_dict(), 'f16.pt')


def sample(t_start=0.99, steps=640):
    cuda = torch.device('cuda:0')
    model = DiT().half().to(cuda)
    model.load_state_dict(torch.load('f16.pt'))
    model.requires_grad_(False).eval()
    x = torch.randn(256, 1, 28, 28, device=cuda)
    t = torch.linspace(t_start, 0, steps + 1, device=cuda)
    for i in tqdm(range(steps)):
        with torch.cuda.amp.autocast():
            eps = model(x, t[i].expand(256)).float()
        alpha = (1 - t[i]) / (1 - t[i + 1])
        mu = (x - (1 - alpha) / t[i].sqrt() * eps) / alpha.sqrt()
        sigma = ((1 - alpha) * t[i + 1] / t[i]).sqrt()
        x = mu + sigma * torch.randn_like(mu)
        if i % 20 == 19:
            img = x.view(16, 16, 28, 28).transpose(1, 2).reshape(448, 448) * 79 + 33
            img = img.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            Image.fromarray(img).save('pv.png')


if __name__ == '__main__':
    train()
    sample()
