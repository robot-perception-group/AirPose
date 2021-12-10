from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from copenet_real.copenet_twoview import copenet_twoview

import os, sys
import is_cluster_mixedmap

def main(args):
    
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning import Callback
    from pytorch_lightning.callbacks import ModelCheckpoint
    import os, sys, time
    os.environ["PYOPENGL_PLATFORM"] = 'egl'
    os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]

    seed_everything(123)

    if args.model.lower() == "copenet_twoview":
        from copenet_real.copenet_twoview import copenet_twoview as copenet_model
    elif args.model.lower() == "copenet_singleview":
        from copenet_real.copenet_singleview import copenet_singleview as copenet_model
    elif args.model.lower() == "hmr":
        from copenet_real.hmr import hmr as copenet_model
    elif args.model.lower() == "muhmr":
        from copenet_real.muhmr import muhmr as copenet_model
    else:
        sys.exit("model not valid")


    # init module
    model = copenet_model(hparams=args)

    # model checkpoint
    ckpt_callback = ModelCheckpoint(monitor="val_loss",save_top_k=1)

    # create logger
    logger = TensorBoardLogger(args.log_dir, name=args.name, version=args.version)

    exp_dir = logger.log_dir

    # cluster callback
    class ClusterCallback(Callback):

        def on_init_start(self,trainer):
            trainer.end_time = time.time() + args.time_to_run

        def on_epoch_end(self,trainer,pl_module):
            # checkpoint and exit to resume later if the job is running for more than desired duration
            if time.time() > trainer.end_time:
                print("running time exceeded. Checkpointing and exiting to resume later")
                trainer.save_checkpoint(os.path.join(exp_dir,"final.ckpt"))
                sys.exit(3)

    if os.path.exists(os.path.join(exp_dir,"final.ckpt")):
        print("pre trained checkpoint found... continuing from there")
        last_ckpt = os.path.join(exp_dir,"final.ckpt")
    else:
        last_ckpt = args.resume_from_checkpoint

    # most basic trainer, uses good defaults
    trainer = Trainer.from_argparse_args(args, 
                                            gpus = 1,
                                            logger = logger,
                                            progress_bar_refresh_rate=100,
                                            resume_from_checkpoint=last_ckpt,
                                            default_root_dir = exp_dir,
                                            checkpoint_callback=ckpt_callback,
                                            callbacks = [ClusterCallback()])

    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(exp_dir,"final.ckpt"))



if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = copenet_twoview.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()
    
    # submit to cluster
    is_cluster_mixedmap.mixedmap(main,
                                [args],
                                verbose=True,
                                max_mem=15,
                                bid_amount=10,
                                use_highend_gpus=True,
                                gpu_count=1,
                                username='nsaini')






# python src/copenet/copenet_trainer_cluster.py --name=test --datapath=/is/cluster/work/nsaini/agora_copenet --batch_size=30 --num_workers=30 --val_steps=1000 --lr=0.00005 --num_epochs=100 --keypoint2d_loss_weight=0.01 --p2d_loss_weight=0.01 --keypoint3d_loss_weight=0 --pose_loss_weight=0 --limbstheta_loss_weight=0 --shape_loss_weight=1000 --trans_loss_weight=10000 --smpltrans_noise_sigma=0.1 --theta_noise_sigma=0 --beta_loss_weight=1000 --time_to_run=3600 