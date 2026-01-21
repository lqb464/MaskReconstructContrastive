from .data.dataset import create_dataloaders_from_folder
from .config.experiment import ExperimentConfig, build_argparser
from .training.utils import get_device, set_seed
from .trainer import Trainer

def main():
    parser = build_argparser()
    args = parser.parse_args()
    cfg = ExperimentConfig.from_args(args)
    
    print("="*100)
    print("Configuration:")
    print(cfg)
    print("="*100)
    
    print("Loss Function:")
    print("Contrastive loss type:", cfg.contrast_loss.contrastive_loss_type)
    print("Contrastive position:", cfg.contrast_loss.contrastive_position)
    print("L =", cfg.training.lambda_contrast, "* L_contrast +", cfg.training.lambda_recon, "* L_recon")
    
    if not cfg.training.enable_reconstruct and not cfg.training.enable_contrastive:
        raise Exception("[Error] Please choose flags for run mode: --enable_reconstruct | --enable_contrastive")

    if cfg.training.enable_contrastive and cfg.training.lambda_contrast == 0:
        raise Exception("[Error] Contrastive training with lambda contrastive = 0")
    
    if cfg.training.enable_reconstruct and cfg.training.lambda_recon == 0:
        raise Exception("[Error] Recontruct training with lambda recontruct = 0")

    if cfg.training.single_view:
        if cfg.training.enable_contrastive:
            raise Exception("[Error] single_view requires --disable-contrastive")
        if cfg.model.enable_saca:
            raise Exception("[Error] single_view does not support SACA; disable SACA or use dual-view")
        if not cfg.training.enable_reconstruct:
            raise Exception("[Error] single_view requires --enable-reconstruct")

    set_seed(cfg.training.seed)
    device = get_device(cfg.training.cpu)

    train_loader, val_loader, _, full_ds = create_dataloaders_from_folder(
        data_root=cfg.data.data_root,
        train_mod=cfg.data.train_mod,
        image_size=cfg.data.image_size,
        plane=cfg.data.plane,
        label_csv=cfg.data.label_csv if cfg.data.label_csv else None,
        label_path_col=cfg.data.label_path_col,
        label_col=cfg.data.label_col,
        batch_size=cfg.training.batch_size,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        seed=cfg.training.seed,
        drop_last=cfg.data.drop_last,
        split_test=cfg.data.split_test,
    )

    print(f"Dataset size: {len(full_ds)}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    trainer = Trainer(cfg, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
