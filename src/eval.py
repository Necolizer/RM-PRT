import hydra
from omegaconf import DictConfig

from trainer import Trainer
"""
    nohup  python src/eval.py common.device=cuda:7 wandb.mode=online >eval.log &
"""
@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.eval()


if __name__ == "__main__":
    main()
