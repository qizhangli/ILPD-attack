import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import update_and_clip, to_np_uint8

__all__ = ["ILPD"]

def hook_ilout(module, input, output):
    module.output=output

def get_hook_pd(ori_ilout, gamma):
    def hook_pd(module, input, output):
        return gamma * output + (1-gamma) * ori_ilout
    return hook_pd

class ILPD(object):
    def __init__(self, args, **kwargs):
        super(ILPD, self).__init__()
        print("ILPD attacking ...")
        self.model = kwargs["source_model"]
        self.coef = args.ilpd_coef
        self.model_name = args.model_name
        self.il_pos = args.ilpd_pos
        self.sigma = args.ilpd_sigma
        self.N = args.ilpd_N
        self._select_pos()
        hook_func = get_hook_pd(0, 1)
        self.hook = self.il_module.register_forward_hook(hook_func)

    def __call__(self, args, ori_img, label, verbose=True):
        adv_img = ori_img.clone()
        for i in range(args.steps):
            input_grad = 0
            for j in range(self.N):
                self._prep_hook(ori_img, i)
                adv_img.requires_grad_(True)
                logits_adv = self.model(adv_img)
                loss = F.cross_entropy(logits_adv, label)
                input_grad += torch.autograd.grad(loss, adv_img)[0].data
            input_grad /= self.N
            adv_img = update_and_clip(ori_img, adv_img, input_grad, args.epsilon, args.step_size, args.constraint)
            if verbose:
                print("Iter {}, loss {:.4f}".format(i, loss.item()))
        return to_np_uint8(adv_img)
    
    def _prep_hook(self, ori_img, iteration):
        if self.sigma == 0 and iteration > 0:
            return
        self.hook.remove()
        with torch.no_grad():
            ilout_hook = self.il_module.register_forward_hook(hook_ilout)
            self.model(ori_img + self.sigma * torch.randn(ori_img.size()).to(ori_img.device))
            ori_ilout = self.il_module.output
            ilout_hook.remove()
        hook_func = get_hook_pd(ori_ilout, self.coef)
        self.hook = self.il_module.register_forward_hook(hook_func)
    
    def _select_pos(self):
        self.model = nn.DataParallel(nn.Sequential(nn.Identity(),
                                                   self.model.module[0],
                                                   self.model.module[1]))
        if self.il_pos == "input":
            self.il_module = eval("self.model.module[0]")
        else:
            self.il_module = eval("self.model.module[2].{}[{}]".format(*self.il_pos.split(".")))
