import torch
import os
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    utility.get_parameter_number(model)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

