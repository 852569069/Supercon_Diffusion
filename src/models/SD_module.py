from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class SD_DDPM_LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        beta1,
        beta2,
        n_T, 
        drop_prob
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()
        for k, v in ddpm_schedules(beta1, beta2, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.drop_prob = drop_prob


    def training_step(self, batch: Any, batch_idx: int):
        # print(np.shape(batch))
        x,c= batch

        x=torch.round(x,decimals=2)

        # get_fumal_input(x)
        int_part = x // 1
        # 使用取余运算符得到小数部分

        dec_part = x % 1
        # 将小数部分乘以10，再取整，得到第一位小数
        first_dec = ((dec_part) * 10) // 1
        # print('first_dec',first_dec[0])
        secend_dec = (((dec_part) * 100) // 1)%10

        
        # print(secend_dec.max(),int_part.max(),first_dec.max())
        # print(secend_dec.min(),int_part.min(),first_dec.min())
        
        x=int_part
        x_f=first_dec
        x_g=secend_dec

        # #转为整数类型
        x=x.type(torch.int64)
        x_f=x_f.type(torch.int64)
        x_g=x_g.type(torch.int64)

        x = torch.nn.functional.one_hot(x, num_classes=10)
        x_f = torch.nn.functional.one_hot(x_f, num_classes=10)
        # print('x_g',x_g[0])
        x_g = torch.nn.functional.one_hot(x_g, num_classes=10)
        # print('x_g',x_g[0:10])

        x=x.type(torch.float32) 
        x_f=x_f.type(torch.float32)
        x_g=x_g.type(torch.float32)

        x=x.unsqueeze(1)
        x_f=x_f.unsqueeze(1)
        x_g=x_g.unsqueeze(1)
        x=torch.cat([x,x_f,x_g],dim=1)

        _ts = torch.randint(1, self.n_T, (x.shape[0],)).cuda() # t ~ Uniform(0, n_T)

        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob)
      
        self.loss=self.criterion(noise, self.net(x_t, _ts / self.n_T,c,context_mask ))

        self.log("train/loss", self.loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return self.loss



    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SD_DDPM_LitModule(None, None, None)
