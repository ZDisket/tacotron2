from torch import nn
from utils import make_non_pad_mask
import torch

class FALGuidedAttentionLoss(torch.nn.Module):
    """Guided attention loss function module.

    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.

    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
        
        Modified for Forced Alignemnt Guided Attention Loss

    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.

        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.

        """
        super(FALGuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self,batch_guided, y_pred):
        guided_masks, xlens, ylens = batch_guided
        _, _, _, alignments = y_pred

        return self.get_loss(alignments,guided_masks,xlens,ylens)
        
    
    def get_loss(self, att_ws,guided_attn_masks, xlens, ylens):
        """Calculate forward propagation.

        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            guided_attn_masks (Tensor): Batch of guided attention weights (B, X, Y)
            xlens (LongTensor): Batch of input lengths (B,).
            ylens (LongTensor): Batch of output lengths (B,).

        Returns:
            Tensor: Guided attention loss value.

        """
        if guided_attn_masks.shape != att_ws.shape:
          print("Attention and masks are diff sizes, skipping")
          return 0.2
          
        if self.masks is None:
            self.masks = self._make_masks(xlens, ylens).to(att_ws.device)
            self.masks = self.masks.transpose(1,2) # masks is orig [y,x] now we make it [x,y]
        
        losses = guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    @staticmethod
    def _make_masks(xlens, ylens):
        """Make masks indicating non-padded part.

        Args:
            xlens (LongTensor or List): Batch of lengths (B,).
            ylens (LongTensor or List): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> xlens, ylens = [5, 2], [8, 5]
            >>> _make_mask(xlens, ylens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)

        """
        in_masks = make_non_pad_mask(xlens)  # (B, T_in)
        out_masks = make_non_pad_mask(ylens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)






class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
