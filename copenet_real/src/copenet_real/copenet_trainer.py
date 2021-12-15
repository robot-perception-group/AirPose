"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from copenet_real.copenet_twoview import copenet_twoview
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


import os, sys, time
os.environ["PYOPENGL_PLATFORM"] = 'egl'
# os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]

# sets seeds for numpy, torch, etc...
# must do for DDP to work well
seed_everything(123)


def main(args):
    
    if args.model.lower() == "copenet_twoview":
        from copenet_real.copenet_twoview import copenet_twoview as copenet_model
    elif args.model.lower() == "copenet_twoview_sep":
        from copenet_real.copenet_twoview_sep import copenet_twoview_sep as copenet_model
    elif args.model.lower() == "copenet_singleview":
        from copenet_real.copenet_singleview import copenet_singleview as copenet_model
    elif args.model.lower() == "hmr":
        from copenet_real.hmr_camswap_difffl import hmr as copenet_model
    elif args.model.lower() == "muhmr":
        from copenet_real.muhmr import muhmr as copenet_model
    elif args.model.lower() == "spin":
        from copenet_real.spin import spin as copenet_model
    else:
        sys.exit("model not valid")

    # init module
    if args.pretrained_checkpoint is not None:
        model = copenet_model.load_from_checkpoint(checkpoint_path=args.pretrained_checkpoint,hparams=args)
    else:
        model = copenet_model(hparams=args)

    # model checkpoint
    ckpt_callback = ModelCheckpoint(monitor="val_loss",save_top_k=1,save_last=True)

    # create logger
    logger = TensorBoardLogger(args.log_dir, name=args.name, version=args.version)

    exp_dir = logger.log_dir
    
    if os.path.exists(os.path.join(exp_dir,"checkpoints","last.ckpt")):
        print("pre trained checkpoint found... continuing from there")
        last_ckpt = os.path.join(exp_dir,"checkpoints","last.ckpt")
    else:
        last_ckpt = args.resume_from_checkpoint

    trainer = Trainer.from_argparse_args(args,
                                            default_root_dir=exp_dir,
                                            gpus = -1,
                                            resume_from_checkpoint=last_ckpt,
                                            checkpoint_callback=ckpt_callback,
                                            callbacks = [ckpt_callback],
                                            logger=logger)
    
    try:
        trainer.fit(model)
        pass
    except KeyboardInterrupt:
        import ipdb; ipdb.set_trace()
        trainer.save_checkpoint(os.path.join(exp_dir,"checkpoints", "last.ckpt"))

    try:
        trainer.save_checkpoint(os.path.join(exp_dir,"checkpoints","last.ckpt"))
    except:
        pass

    import ipdb; ipdb.set_trace()
    # res = trainer.test()
    # res["fig"].write_html(os.path.join("copenet_logs",args.name,"fig.html"))


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = copenet_twoview.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()
    # import ipdb; ipdb.set_trace()
    main(args)
