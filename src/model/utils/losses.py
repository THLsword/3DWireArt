import torch
from torch import nn

class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, preds, cls_labels):
        loss = self.loss_fn(preds, cls_labels)
        return loss

class ConsistencyLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, tgt):
        loss_list = []
        for i in range(len(tgt)):
            for j in range(len(tgt)):
                if i < j:
                    loss_list.append(self.loss_fn(tgt[i], tgt[j]))
                else:
                    continue
        return sum(loss_list) / len(tgt)

class MSELoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, rendered_imgs, true_color_imgs, GT_imgs):
        mask = (true_color_imgs == 0)
        mask = mask.all(dim=-1) # （B,4,256,256)
        rendered_imgs = rendered_imgs.sum(dim=-1,keepdim=False)/3 # （B,4,256,256)
        GT_imgs = GT_imgs.sum(dim=-1,keepdim=False)/3 # （B,4,256,256)
        masked_rendered_imgs = rendered_imgs[mask]
        masked_GT_imgs = GT_imgs[mask]
        # hook = masked_rendered_imgs.register_hook(lambda grad: print("masked_rendered_imgs.grad: ", grad))

        # test: onle one view
        t_mask = mask[:,0:1,:,:]
        t_rendered_imgs = rendered_imgs[:,0:1,:,:]
        t_GT_imgs = GT_imgs[:,0:1,:,:]
        t_masked_rendered_imgs = t_rendered_imgs[t_mask]
        # hook = t_masked_rendered_imgs.register_hook(lambda grad: print("t_masked_rendered_imgs.grad: ", grad))
        t_masked_GT_imgs = t_GT_imgs[t_mask]

        # loss = self.loss_fn(t_masked_rendered_imgs, t_masked_GT_imgs) # -> one view test
        loss = self.loss_fn(masked_rendered_imgs, masked_GT_imgs) # -> normal trainning

        # print("loss")
        # print("t_masked_rendered_imgs.r_grad: ", t_masked_rendered_imgs.requires_grad)
        
        return loss

class WeightL1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.sum(dim=-1)/3     # (B,4,256,256,3) -> (B,4,256,256)
        target = target.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256)
        # L1 loss
        # count_all = torch.sum(target > 0)
        # L1_loss = torch.abs(pred - target)
        # L1_loss = L1_loss.sum()/count_all

        # count_all = torch.sum((pred <= target) & (target > 0))
        count_all = torch.sum(target > 0)
        L1_loss = torch.where(pred <= target, (pred - target)*4, pred - target)
        # L1_loss = pred - target
        L1_loss = L1_loss.abs()
        L1_loss = L1_loss.sum()/count_all

        return L1_loss

def Brightness_loss(pred, target):
    pred = pred.sum(dim=-1)/3     # (B,4,256,256,3) -> (B,4,256,256)
    target = target.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256)
    # compute average brightness of each img
    mask = target > 0
    count_each = mask.sum(dim = [2,3], keepdim = True) # (B,4,1,1)
    count_all = torch.sum(target > 0)
    avg_value = target.sum(dim=[2,3], keepdim = True) / count_each # (B,4,1,1)
    # compute loss
    temp = torch.where(pred > 0, pred - avg_value, pred)
    brightness_loss = temp.abs().sum()/count_all

    return brightness_loss

class BinaryCrossEntrophy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction='none')

    def forward(self, predict, target, true_outputs):
        # BCE loss要攤平成兩個維度[B,-1],不然會報錯
        B = target.shape[0]
        predict = predict.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256,1)
        predict = predict.view(B, -1) # torch.Size([8, 262144])

        target = target.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256,1)
        target = target.view(B, -1)

        true_outputs = true_outputs.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256,1)
        true_outputs = true_outputs.view(B, -1)

        mask = true_outputs > 0
        A_masked = torch.where(mask, predict, torch.zeros_like(predict))
        B_masked = torch.where(mask, target, torch.zeros_like(target))

        # 计算BCE损失
        loss = self.loss_fn(A_masked, B_masked)
        mask_loss = torch.where(mask, loss, torch.zeros_like(loss))

        # 求平均损失（只考虑mask为true的部分）
        average_loss = torch.sum(mask_loss) / torch.sum(mask)
        
        return average_loss