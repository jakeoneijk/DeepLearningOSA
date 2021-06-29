import os
from Trainer.LogWriter import LogWriter
from HParams import HParams
from abc import ABC, abstractmethod
from enum import Enum,unique
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

@unique
class TrainState(Enum):
    TRAIN = "train"
    VALIDATE = "valid"
    TEST = "test"
 
class Trainer(ABC):
    def __init__(self,model,h_params:HParams):
        self.h_params = h_params

        self.model = model.to(h_params.resource.device)
        self.set_optimizer()
        self.criteria = {}
        self.set_criteria()
        self.set_lr_scheduler()

        self.to_h_params_device()

        self.train_data_loader = None
        self.valid_data_loader = None
        self.test_data_loader = None

        self.seed = (int)(torch.cuda.initial_seed() / (2**32)) if self.h_params.train.seed is None else self.h_params.train.seed
        
        self.set_seeds(self.h_params.train.seed_strict)

        self.check_point_num = 0 #binary
        self.current_epoch = 0
        self.total_epoch = self.h_params.train.epoch

        self.best_valid_metric = None
        self.best_valid_epoch = 0

        self.global_step = 0
        self.local_step = 0

        self.log_writer = LogWriter(self.h_params)

    '''
    ==============================================================
    abstract method start
    ==============================================================
    '''

    @abstractmethod
    def set_criteria(self):
        '''
        set self.criteria
        self.criteria[key] = loss
        '''
        raise NotImplementedError

    @abstractmethod
    def set_optimizer(self):
        '''
        set self.optimizer
        '''
        raise NotImplementedError
    
    @abstractmethod
    def set_lr_scheduler(self):
        '''
        set self.lr_scheduler
        it can be None
        '''
        raise NotImplementedError

    @abstractmethod
    def get_metric_name_list(self):
        """
        return name list of metric
        ex)loss_name_list = ["total loss"]
        """
        raise NotImplementedError
    
    @abstractmethod
    def run_step(self,data,metric):
        """
        run 1 step
        1. get data
        2. use model
        3. calculate loss
        4. put the loss in metric (append)
        return loss,metric
        """
        raise NotImplementedError

    @abstractmethod
    def lr_scheduler_step(self):
        '''
        lr sheduler
        ex)self.lr_scheduler.step()
        '''
        raise NotImplementedError

    @abstractmethod
    def save_best_model(self,prev_best_metric, current_metric):
        """
        compare what is the best metric
        If current_metric is better, 
            1.save best model
            2. self.best_valid_epoch = self.current_epoch
        Return
            better metric
        """
        raise NotImplementedError
    
    @abstractmethod
    def log_metric(self, metrics ,data_size: int,train_state=TrainState.TRAIN):
        """
        log and tensorboard log
        """
        raise NotImplementedError

    '''
    ==============================================================
    abstract method end
    ==============================================================
    '''
    def to_h_params_device(self):
        self.model = self.model.to(self.h_params.resource.device)
        for criteria_key in self.criteria:
            self.criteria[criteria_key] = self.criteria[criteria_key].to(self.h_params.resource.device)
    
    def set_seeds(self,strict=False):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            if strict:
                torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def set_data_loader(self,train,valid,test):
        self.train_data_loader = train
        self.valid_data_loader = valid
        self.test_data_loader = test
    
    def fit(self,use_val_metric=True):
        
        for _ in range(self.current_epoch, self.total_epoch):
            self.log_writer.print_and_log(f'----------------------- Start epoch : {self.current_epoch} / {self.h_params.train.epoch} -----------------------',self.global_step)
            self.log_writer.print_and_log(f'current best epoch: {self.best_valid_epoch}',self.global_step)
            self.log_writer.print_and_log(f'-------------------------------------------------------------------------------------------------------',self.global_step)
    
            #Train
            self.log_writer.print_and_log('train_start',self.global_step)
            train_metric = self.run_epoch(self.train_data_loader,TrainState.TRAIN)
            
            #Valid
            self.log_writer.print_and_log('valid_start',self.global_step)
            with torch.no_grad():
                valid_metric = self.run_epoch(self.valid_data_loader,TrainState.VALIDATE)
            
            self.best_valid_metric = self.save_best_model(self.best_valid_metric, valid_metric)
            
            self.current_epoch += 1

        #Test   
        self.log_writer.print_and_log(f'test_best_epoch: {self.best_valid_epoch}',self.global_step)
        self.load_module()
        with torch.no_grad():
            test_metric = self.run_epoch(self.test_data_loader,TrainState.TEST)

        self.final_report(test_metric)
        print("Training complete")
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState):
        if train_state == TrainState.TRAIN:
            self.model.train()
        else:
            self.model.eval()
        dataset_size = len(dataloader)
        metric = self.metric_init()

        for step,data in enumerate(dataloader):
            self.local_step = step
            loss,metric = self.run_step(data,metric)
        
            if train_state == TrainState.TRAIN:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.local_step % self.h_params.log.log_every_local_step == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                    
                self.global_step += 1

                if self.lr_scheduler is not None:
                    self.lr_scheduler_step()
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)

        if train_state == TrainState.TRAIN:
            self.save_checkpoint(self.check_point_num)
            self.check_point_num = int((self.check_point_num+1)%2)

        return metric
    
    def metric_init(self):
        loss_name_list = self.get_metric_name_list()
        initialized_metric = dict()

        for loss_name in loss_name_list:
            initialized_metric[loss_name] = np.array([])

        return initialized_metric

    def save_module(self,name="load",prefix='model'):
        path = os.path.join(self.h_params.log.model_save_path,f'{prefix}_{name}.pth')
        torch.save(self.model.state_dict(), path)

    def load_module(self,name="load",prefix='model'):
        path = os.path.join(self.h_params.log.model_save_path,f'{prefix}_{name}.pth')
        best_model_load = torch.load(path)
        self.model.load_state_dict(best_model_load)
    
    def save_checkpoint(self,prefix=""):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'models': self.model.state_dict(),
            'optimizers': self.optimizer.state_dict(),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }
        path = os.path.join(self.h_params.log.model_save_path,f'{self.model.__class__.__name__}_checkpoint{prefix}.pth')
        torch.save(train_state,path)

    def resume(self,filename:str):
        cpt = torch.load(filename)
        self.seed = cpt['seed']
        self.set_seeds(self.h_params.train.seed_strict)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']
        self.model.load_state_dict(cpt['models'])
        self.optimizer.load_state_dict(cpt['optimizers'])
        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']