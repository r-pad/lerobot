import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from copy import deepcopy
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler
from multiprocessing import set_start_method
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

# for loading high-level policy only
# from dp3_workspace_for_loading_ckpt import DP3WorkspaceHighLevel

OmegaConf.register_new_resolver("eval", eval, replace=True)

def ddp_setup():
    os.environ["NCCL_P2P_LEVEL"] = "NVL"
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None, pretrained_goal_model=None):
        self.cfg = cfg
        # print("cfg: ", cfg)
        # input("Press Enter to continue...")
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)

        if cfg.load_policy_path is not None:
            # payload = torch.load(cfg.load_policy_path.open('rb'), pickle_module=dill, map_location='cpu')
            # self.model.load_state_dict(payload['state_dicts']['model'])
            self.load_policy(cfg.load_policy_path)
            
        self.model.train()

        self.pretrained_goal_model = pretrained_goal_model
        if cfg.training.pretrained_weighted_displacement_goal_model is not None:
            from test_PointNet2.model_invariant import PointNet2_small2, PointNet2, PointNet2_super
            if "LOCAL_RANK" in os.environ:
                device = torch.device(int(os.environ["LOCAL_RANK"]))
            else:
                device = torch.device("cuda")
            pointnet2_model = PointNet2_super(num_classes=13).to(device)
            pointnet2_model.load_state_dict(torch.load(cfg.training.pretrained_weighted_displacement_goal_model))
            pointnet2_model.eval()
            self.pretrained_weighted_displacement_goal_model = pointnet2_model
            

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        # self.amp_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.training.use_amp)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1000
            cfg.training.val_every = 20
            # cfg.training.sample_every = 1
            cfg.dataloader.batch_size = 32
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        # RUN_VALIDATION = False # reduce time cost
        RUN_VALIDATION = True # reduce time cost
        
        # resume training
        # if cfg.training.resume:
        #     print("Resuming from checkpoint")
        #     lastest_ckpt_path = self.get_checkpoint_path()
        #     if lastest_ckpt_path.is_file():
        #         print(f"Resuming from checkpoint {lastest_ckpt_path}")
        #         self.load_checkpoint(path=lastest_ckpt_path)
        # if cfg.load_checkpoint_path is not None:
        #     print(f"Resuming from checkpoint {cfg.load_checkpoint_path}")
        #     self.load_checkpoint(path=cfg.load_checkpoint_path)
        

        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)

        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, 
                                      shuffle=False,
                                      sampler=DistributedSampler(dataset),
                                      batch_size=cfg.dataloader.batch_size,
                                      num_workers=cfg.dataloader.num_workers,
                                      pin_memory=True,
                                      )

        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        if cfg.training.use_dataset_normalization:
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        if cfg.task.run_eval_rollout:
            env_runner: BaseRunner
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)

            if env_runner is not None:
                assert isinstance(env_runner, BaseRunner)
        else:
            env_runner = None

        cprint("==================================", "green")
        cprint("================ SAVE DIR ================", "green")
        cprint(f"{self.output_dir}", "green")
        cprint("==================================", "green")
        
        cfg.logging.name = str(cfg.logging.name)
        cprint("-----------------------------", "yellow")
        cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
        cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
        cprint("-----------------------------", "yellow")
        # configure logging
        if os.environ['LOCAL_RANK'] == '0':
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                }
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        device = torch.device(self.gpu_id)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)
        
        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        # if self.ema_model is not None:
        #     self.ema_model = DDP(self.ema_model, device_ids=[self.gpu_id])

        # save batch for sampling
        train_sampling_batch = None


        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        for local_epoch_idx in range(cfg.training.num_epochs):
            train_dataloader.sampler.set_epoch(self.epoch)
            b_sz = cfg.dataloader.batch_size
            print(f"[GPU{self.gpu_id}] Epoch {self.epoch} | Batchsize: {b_sz} | Steps: {len(train_dataloader)}")
            
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # print("train batch_idx {}/{}".format(batch_idx, len(tepoch)))
                    t1 = time.time()
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    if self.pretrained_goal_model is not None and np.random.random() < 0.5:
                        with torch.no_grad():
                            goal_model_output = self.pretrained_goal_model.predict_action(batch['obs'])
                        goal_model_output = dict_apply(goal_model_output, lambda x: x.to(device))
                        goal_model_output = goal_model_output['action']
                        
                        reshaped_goal_model_output = goal_model_output[:, :2, :].reshape((-1, 2, 4, 3))

                        batch['obs']['goal_gripper_pcd'] = reshaped_goal_model_output                    
                                
                        
                    # compute loss
                    t1_1 = time.time()
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.training.use_amp):
                        raw_loss, loss_dict = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()
                    # self.amp_scaler.scale(loss).backward()    
                    
                    t1_2 = time.time()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        
                    t1_3 = time.time()
                    # update ema
                    if cfg.training.use_ema:
                        # ema.step(self.model)
                        ema.step(self.model.module)
                    t1_4 = time.time()
                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    t1_5 = time.time()
                    step_log.update(loss_dict)
                    t2 = time.time()
                    
                    if verbose:
                        print(f"total one step time: {t2-t1:.3f}")
                        print(f" compute loss time: {t1_2-t1_1:.3f}")
                        print(f" step optimizer time: {t1_3-t1_2:.3f}")
                        print(f" update ema time: {t1_4-t1_3:.3f}")
                        print(f" logging time: {t1_5-t1_4:.3f}")

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if os.environ['LOCAL_RANK'] == '0':
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

                    # import pdb; pdb.set_trace()
                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval for this epoch ==========
            policy = self.model
            if cfg.training.use_ema:
                policy = self.ema_model
            policy.eval()

            # run rollout
            if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and os.environ['LOCAL_RANK'] == '0':
                # first checkpointing then running the eval
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint()
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()
                
                if self.epoch == 0 and not cfg.eval_first:
                    pass
                elif env_runner is None:
                    pass
                else:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(cfg, policy, self.epoch)
                    # wandb_run.log(runner_log, step=self.epoch)
                    t4 = time.time()
                    cprint(f"rollout time: {t4-t3:.3f}", "red")
                    # log all
                    step_log.update(runner_log)

                # TODO: add dagger here
                # 1. should store the final state in env_runner.run
                # 2. judge based on the opened door angles -- if it is below a certain threshold, should rerun demonstration generation code on it
                # 3. add the new demonstration to the dataset & dataloader. 
                
            # run validation
            if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                print("Validation epoch {}/{}".format(self.epoch, cfg.training.num_epochs))
                with torch.no_grad():
                    val_losses = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                            if self.pretrained_goal_model is not None:
                                with torch.no_grad():
                                    goal_model_output = self.pretrained_goal_model.predict_action(batch['obs'])
                                goal_model_output = dict_apply(goal_model_output, lambda x: x.to(device))
                                goal_model_output = goal_model_output['action']
                                
                                reshaped_goal_model_output = goal_model_output[:, :2, :].reshape((-1, 2, 4, 3))

                                batch['obs']['goal_gripper_pcd'] = reshaped_goal_model_output

                            loss, loss_dict = self.model(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                    if len(val_losses) > 0:
                        val_loss = torch.mean(torch.tensor(val_losses)).item()
                        # log epoch average validation loss
                        step_log['val_loss'] = val_loss

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    # sample trajectory from training set, and evaluate difference
                    
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    obs_dict = batch['obs']
                    cat_idx = batch['cat_idx'].squeeze(-1)
                    if self.cfg.policy.prediction_target == 'action':
                        gt_action = batch['action']
                    else:
                        gt_action = batch['obs'][self.cfg.policy.prediction_target].flatten(start_dim=2)
                    
                    result = policy.predict_action(obs_dict, cat_idx)
                    pred_action = result['action_pred']
                    mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del obs_dict
                    del gt_action
                    del result
                    del pred_action
                    del mse

            if env_runner is None:
                step_log['test_mean_score'] = - train_loss
                
            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt: 
                if self.epoch == 0 and not cfg.eval_first:
                    pass
                else:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if self.epoch % 10 == 0 and self.epoch > 0:
                        self.save_checkpoint(tag=f'epoch-{self.epoch}')
                    # if cfg.checkpoint.save_last_snapshot:
                    #     self.save_snapshot()

                    if 'test_mean_score' in step_log:
                        self.save_checkpoint(tag=f'epoch-{self.epoch}-test_mean_score-{step_log["test_mean_score"]:.3f}')
                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items() :
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                        
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                    else:
                        self.save_checkpoint(tag=f'epoch-{self.epoch}')
                    
                    
                        
            # ========= eval end for this epoch ==========
            policy.train()

            # end of epoch
            # log of last step is combined with validation and rollout
            if os.environ['LOCAL_RANK'] == '0':
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            del step_log

    def eval(self):
        # load the latest checkpoint
        
        cfg = copy.deepcopy(self.cfg)
        
        lastest_ckpt_path = self.get_checkpoint_path(tag="latest")
        if lastest_ckpt_path.is_file():
            cprint(f"Resuming from checkpoint {lastest_ckpt_path}", 'magenta')
            self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseRunner)
        policy = self.model
        if cfg.training.use_ema:
            policy = self.ema_model
        policy.eval()
        policy.cuda()

        runner_log = env_runner.run(policy)
        
      
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        # import pdb; pdb.set_trace()
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        if key == 'model' and isinstance(value, DDP):
                            payload['state_dicts'][key] = value.module.state_dict()
                        else:
                            payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)

        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        
        # cprint("saving checkpoint path {}".format(str(path.absolute())), 'red')
        # policy = self.ema_model
        # policy.eval()
        # train_sampling_batch = torch.load("/mnt/RoboGen_sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/tmp_obs.pkl")
        # batch = dict_apply(train_sampling_batch, lambda x: torch.from_numpy(x).unsqueeze(0).to(torch.device("cuda"), non_blocking=True))
        # obs_dict = batch
        # gt_action = batch['goal_gripper_pcd'][:, 0, :, :]
        # result = policy.predict_action(obs_dict, pdb=True)
        
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
    def load_policy(self, path):
        path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        if 'LOCAL_RANK' in os.environ:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            device = torch.device(self.gpu_id)
        else:
            device = torch.device("cuda")
            
        cprint("loading policy from {}".format(str(path.absolute())), "green")
        self.model = hydra.utils.instantiate(self.cfg.policy)
        self.model.load_state_dict(payload['state_dicts']['model'])
        self.model.to(device)
        
        self.ema_model.load_state_dict(payload['state_dicts']['ema_model'])
    

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()
        exclude_keys += self.cfg.load_exclude_keys
        
        if 'LOCAL_RANK' in os.environ:
            self.gpu_id = int(os.environ["LOCAL_RANK"])
            device = torch.device(self.gpu_id)
        else:
            device = torch.device("cuda")
        
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                print(f"loading {key}")
                if key == 'optimizer': continue
                if key == 'model':
                    self.model = hydra.utils.instantiate(self.cfg.policy)
                    self.model.load_state_dict(value, **kwargs)
                    self.model.to(device)
                else:
                    self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    

        # self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
        # self.model.to(device)
        # optimizer_to(self.optimizer, device)
        # self.ema_model.to(device)
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=['pretrained_goal_model', "pretrained_weighted_displacement_goal_model", "amp_scaler"], 
            include_keys=None, 
            **kwargs):
        cprint(f"Loading checkpoint from {path}", "green")
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def initialize_pretrained_high_level_policy():
    # configure model
    # on autobot
    goal_exp_dir = '/home/chenyuan/Robogen-sim2real/3d_diffusion_policy/3D-Diffusion-Policy/3D-Diffusion-Policy/data/0807-200-obj-pred-goal-gripper-PointNet2-backbone-UNet-diffusion-ep-75-epsilon/2024.08.07/14.03.40_train_dp3_robogen_open_door'
    # on robocluster
    # goal_exp_dir = '/ocean/projects/cis240052p/ckuo1/RoboGen-sim2real/pretrained_high-level_policy/14.03.40_train_dp3_robogen_open_door'

    goal_checkpoint_name = 'epoch-30.ckpt'
    goal_checkpoint_path = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)

    with hydra.initialize(config_path='diffusion_policy_3d/config'):  # same config_path as used by @hydra.main
        recomposed_config = hydra.compose(
            config_name="dp3.yaml",  # same config_name as used by @hydra.main
            overrides=OmegaConf.load("{}/.hydra/overrides.yaml".format(goal_exp_dir)),
        )
    goal_cfg = recomposed_config

    goal_workspace = TrainDP3Workspace(goal_cfg)
    goal_checkpoint_dir = "{}/checkpoints/{}".format(goal_exp_dir, goal_checkpoint_name)
    goal_workspace.load_checkpoint(path=goal_checkpoint_dir)

    goal_policy = deepcopy(goal_workspace.model)
    if goal_workspace.cfg.training.use_ema:
        goal_policy = deepcopy(goal_workspace.ema_model)
    goal_policy.eval()
    goal_policy.reset()
    device = torch.device(int(os.environ["LOCAL_RANK"]))
    goal_policy = goal_policy.to(device)
    pretrained_goal_model = goal_policy  # Assuming goal_policy is defined in your scope
    return pretrained_goal_model
# pretrained_goal_model = initialize_pretrained_high_level_policy()
import sys
sys.path.append('/home/chenyuan/RoboGen-sim2real')
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):

    ddp_setup()
    if cfg.use_pretrained_high_level_policy_as_low_level_input:
        cprint(f'=====================================================================================================', 'green')
        cprint(f'Using {cfg.use_pretrained_high_level_policy_as_low_level_input}', 'green')
        cprint(f'=====================================================================================================', 'green')
        
        #############################################
        # for loading pre-trained high-level policy #
        #############################################        
        workspace = TrainDP3Workspace(cfg, pretrained_goal_model=pretrained_goal_model)
    else :
        workspace = TrainDP3Workspace(cfg, pretrained_goal_model=None)
    workspace.run()
    destroy_process_group()

if __name__ == "__main__":

    main()
    