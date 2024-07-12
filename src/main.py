import hydra
from omegaconf import DictConfig

from trainer import Trainer
"""
    HYDRA_FULL_ERROR=1 nohup  python src/main.py common.device=cuda:5 wandb.mode=offline >train.log &
"""
@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
