from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import numpy as np
import random
# from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm
import wandb
import logging
from agent import Agent
from collector import Collector
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from make_reconstructions import make_reconstructions_from_batch
from models.actor_critic import ActorCritic
from models.world_model import WorldModel, WorldModelPredictive
from utils import configure_optimizer, EpisodeDirManager, set_seed, LR_Scheduler
from tester import Tester

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if True: # not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            # shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            # shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            # shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)


        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        init_seed(self.cfg.common.seed)
        if self.cfg.training.should:
            feeder = instantiate(cfg.datasets.train)
            self.train_dataset = DataLoader(
                dataset=feeder,
                batch_size=self.cfg.datasets.batch_size,
                shuffle=True,
                num_workers=self.cfg.datasets.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        if self.cfg.evaluation.should:
            feeder = instantiate(cfg.datasets.test)
            self.test_dataset = DataLoader(
                dataset=feeder,
                batch_size=self.cfg.datasets.batch_size,
                shuffle=False,
                num_workers=self.cfg.datasets.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

        assert self.cfg.training.should or self.cfg.evaluation.should
        # env = train_env if self.cfg.training.should else test_env

        if self.cfg.use_origin_img:
            # tokenizer = instantiate(cfg.tokenizer)
            # target_tokenizer = instantiate(cfg.tokenizer)
            # target_tokenizer.load_state_dict(target_tokenizer.state_dict())

            # predictor = instantiate(cfg.predictor)
            # actor = instantiate(cfg.actor)
            model = instantiate(cfg.model) # RT1_state(action_nums=3, bins=50)
            self.agent = Agent(model=model,loss_weight=self.cfg.training.agent.loss_weight,use_origin_img=self.cfg.use_origin_img).to(self.device)
            
            # self.agent = Agent(tokenizer=tokenizer,target_tokenizer=target_tokenizer,actor=actor,predictor=predictor,loss_weight=self.cfg.training.agent.loss_weight,use_origin_img=self.cfg.use_origin_img).to(self.device)
            print(f'{sum(p.numel() for p in self.agent.parameters())} parameters in agent')
            try:
                print(f'{sum(p.numel() for p in self.agent.model.embedder.parameters())} parameters in agent.embedder')
            except:
                pass
            self.optimizer_agent = torch.optim.AdamW(self.agent.parameters(), lr=cfg.training.agent.learning_rate, weight_decay=cfg.training.agent.weight_decay)
            self.lr_scheduler_agent=LR_Scheduler[cfg.training.agent.lr_scheduler_func](self.optimizer_agent,**cfg.training.agent.lr_scheduler_config)
        else:
            tokenizer = instantiate(cfg.tokenizer)
            world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=cfg.env.num_actions, state_size=cfg.env.num_states, config=instantiate(cfg.world_model), loss_weight = cfg.training.world_model.loss_weight)
            # actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=env.num_actions)
            self.agent = Agent(tokenizer,world_model).to(self.device)
            print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
            print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
            
            self.optimizer_tokenizer = torch.optim.Adam(self.agent.tokenizer.parameters(), lr=cfg.training.tokenizer.learning_rate)
            self.optimizer_world_model = configure_optimizer(self.agent.world_model, cfg.training.world_model.learning_rate, cfg.training.world_model.weight_decay,'embedder.text_embedder.null_text_embed')
            self.lr_scheduler_world_model=LR_Scheduler[cfg.training.world_model.lr_scheduler_func](self.optimizer_world_model,**cfg.training.world_model.lr_scheduler_config)
            
        if cfg.common.resume:
            self.agent.load(**cfg.initialization, device=self.device)

    def run(self) -> None:
        min_loss = None
        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []
            save_should = False
            if self.cfg.training.should:
                to_log += self.train_agent(epoch)

            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                to_log += self.eval_agent(epoch)
                if min_loss is None or min_loss>to_log[-1]['agent/eval/total_loss']:
                    min_loss = to_log[-1]['agent/eval/total_loss']
                    save_should = True
            if self.cfg.training.should and save_should:
                self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

            to_log.append({'duration': (time.time() - start_time) / 3600})
            for metrics in to_log:
                wandb.log({'epoch': epoch, **metrics})
            # print(to_log)
            logging.info(to_log)
        self.finish()
    
    def eval(self) -> None:

         
        Tester(self.agent,self.cfg,self.episode_dir)
        # for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):

        #     print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
        #     start_time = time.time()
        #     to_log = []

        #     if self.cfg.training.should:
        #         to_log += self.train_agent(epoch)

        #     if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
        #         to_log += self.eval_agent(epoch)

        #     if self.cfg.training.should:
        #         self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

        #     to_log.append({'duration': (time.time() - start_time) / 3600})
        #     for metrics in to_log:
        #         wandb.log({'epoch': epoch, **metrics})

        self.finish()
        
    def train_agent(self, epoch: int) -> None:
        self.agent.train()
        self.agent.zero_grad()

        metrics_tokenizer, metrics_world_model, metrics_actor = {}, {}, {}

        cfg_agent = self.cfg.training.agent
        steps_per_epoch = len(self.train_dataset)

        if epoch > cfg_agent.start_after_epochs:
            metrics_agent = self.train_component(self.agent, self.optimizer_agent, steps_per_epoch=steps_per_epoch, lr_scheduler=self.lr_scheduler_agent, **cfg_agent)
        self.agent.eval()

        return [{'epoch': epoch, **metrics_agent}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, steps_per_epoch: int,  max_grad_norm: Optional[float],  lr_scheduler= None, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)
        for _, (imgs, instr, actions, states_tensor, next_imgs, index) in enumerate(tqdm(self.train_dataset, desc="Training", ncols=100)):
            """batch['observation'] is supposed to be channels first and in [0, 1]"""
            B, SEQ, F, H, W, C = imgs.shape
            imgs = imgs.contiguous().view(B*SEQ, F, H, W, C).float()
            next_imgs = next_imgs.contiguous().view(B*SEQ, H, W, C).float()
            _, _, V = actions.shape
            B, SEQ, F, V2 = states_tensor.shape
            actions = actions.contiguous().view(-1, V).float()
            states_tensor = states_tensor.contiguous().view(-1, F, V2).float()

            imgs = imgs.permute(0, 1, 4, 2, 3) # 'b f c h w'
            next_imgs = next_imgs.permute(0, 3, 1, 2).unsqueeze(dim=1)
            instructions = [] 
            for i in instr:
                instructions += [i] * SEQ
            batch=dict()
            batch['observations']=imgs
            batch['next_observations']=next_imgs
            batch['states']=states_tensor
            batch['actions']=actions
            batch['instr']=instructions
            optimizer.zero_grad()
            batch = self._to_device(batch)

            losses = component.compute_loss(batch, **kwargs_loss) 
            loss_total_step = losses.loss_total
            loss_total_step.backward()
            loss_total_epoch += loss_total_step.item() / steps_per_epoch

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/train/{loss_name}"] += loss_value / steps_per_epoch

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        metrics = {f'{str(component)}/train/total_loss': loss_total_epoch, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> None:
        self.agent.eval()

        metrics_tokenizer, metrics_world_model,metrics_actor = {}, {}, {}

        cfg_agent = self.cfg.evaluation.agent

        if epoch > cfg_agent.start_after_epochs:
            metrics_agent = self.eval_component(self.agent)

        return [metrics_agent]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, **kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        steps = 0
        pbar = tqdm(desc=f"Evaluating {str(component)}", file=sys.stdout)
        for _, (imgs, instr, actions, states_tensor, next_imgs, index) in enumerate(tqdm(self.test_dataset, desc="Testing", ncols=100)):
            """batch['observation'] is supposed to be channels first and in [0, 1]"""
            B, SEQ, F, H, W, C = imgs.shape
            imgs = imgs.contiguous().view(B*SEQ, F, H, W, C).float()
            next_imgs = next_imgs.contiguous().view(B*SEQ, H, W, C).float()
            _, _, V = actions.shape
            B, SEQ, F, V2 = states_tensor.shape
            actions = actions.contiguous().view(-1, V).float()
            states_tensor = states_tensor.contiguous().view(-1, F, V2).float()

            imgs = imgs.permute(0, 1, 4, 2, 3)
            next_imgs = next_imgs.permute(0, 3, 1, 2).unsqueeze(dim=1)
            instructions = [] 
            for i in instr:
                instructions += [i] * SEQ

            batch=dict()
            batch['observations']=imgs
            batch['next_observations']=next_imgs
            batch['actions']=actions
            batch['states']=states_tensor
            batch['instr']=instructions
            batch = self._to_device(batch)
            losses = component.compute_loss(batch, **kwargs_loss)
            loss_total_epoch += losses.loss_total.item()

            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{str(component)}/eval/{loss_name}"] += loss_value

            steps += 1
            pbar.update(1)
        intermediate_losses = {k: v / steps for k, v in intermediate_losses.items()}
        metrics = {f'{str(component)}/eval/total_loss': loss_total_epoch / steps, **intermediate_losses}
        return metrics

    @torch.no_grad()
    def inspect_imagination(self, epoch: int) -> None:
        mode_str = 'imagination'
        batch = self.test_dataset.sample_batch(batch_num_samples=self.episode_manager_imagination.max_num_episodes, sequence_length=1 + self.cfg.training.actor_critic.burn_in, sample_from_start=False)
        outputs = self.agent.actor_critic.imagine(self._to_device(batch), self.agent.tokenizer, self.agent.world_model, horizon=self.cfg.evaluation.actor_critic.horizon, show_pbar=True)

        to_log = []
        for i, (o, a, r, d) in enumerate(zip(outputs.observations.cpu(), outputs.actions.cpu(), outputs.rewards.cpu(), outputs.ends.long().cpu())):  # Make everything (N, T, ...) instead of (T, N, ...)
            episode = Episode(o, a, r, d, torch.ones_like(d))
            episode_id = (epoch - 1 - self.cfg.training.actor_critic.start_after_epochs) * outputs.observations.size(0) + i
            self.episode_manager_imagination.save(episode, episode_id, epoch)

            metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
            metrics_episode['episode_num'] = episode_id
            metrics_episode['action_histogram'] = wandb.Histogram(episode.actions.numpy(), num_bins=self.agent.world_model.act_vocab_size)
            to_log.append({f'{mode_str}/{k}': v for k, v in metrics_episode.items()})

        return to_log

    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        torch.save(self.agent.state_dict(), self.ckpt_dir / 'last.pt')
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save({
                # "optimizer_tokenizer": self.optimizer_tokenizer.state_dict() if self.optimizer_tokenizer is not None else None,
                "optimizer_agent": self.optimizer_agent.state_dict(),
                "lr_scheduler_agent":self.lr_scheduler_agent.state_dict()
                # "optimizer_actor_critic": self.optimizer_actor_critic.state_dict(),
            }, self.ckpt_dir / 'optimizer.pt')
            # ckpt_dataset_dir = self.ckpt_dir / 'dataset'
            # ckpt_dataset_dir.mkdir(exist_ok=True, parents=False)
            # self.train_dataset.update_disk_checkpoint(ckpt_dataset_dir)
            # if self.cfg.evaluation.should:
            #     torch.save(self.test_dataset.num_seen_episodes, self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')

    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)

    def load_checkpoint(self) -> None:
        assert self.ckpt_dir.is_dir()
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        self.agent.load(self.ckpt_dir / 'last.pt', device=self.device)
        ckpt_opt = torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device)
        self.optimizer_tokenizer.load_state_dict(ckpt_opt['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(ckpt_opt['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(ckpt_opt['optimizer_actor_critic'])
        self.train_dataset.load_disk_checkpoint(self.ckpt_dir / 'dataset')
        if self.cfg.evaluation.should:
            self.test_dataset.num_seen_episodes = torch.load(self.ckpt_dir / 'num_seen_episodes_test_dataset.pt')
        print(f'Successfully loaded model, optimizer and {len(self.train_dataset)} episodes from {self.ckpt_dir.absolute()}.')

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.device) if torch.is_tensor(batch[k]) else batch[k] for k in batch}

    def finish(self) -> None:
        wandb.finish()
