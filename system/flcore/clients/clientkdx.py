import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from flcore.clients.helper_function import ContrastiveLoss , HardTripletLoss
# import scipy.linalg as la
# from flcore.clients.helper_function import
# from scipy.sparse.linalg import svds
class clientKDX(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mentee_learning_rate = args.mentee_learning_rate

        self.global_model = copy.deepcopy(args.model)
        self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.mentee_learning_rate)
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g,
            gamma=args.learning_rate_decay_gamma
        )

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)
        self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_W,
            gamma=args.learning_rate_decay_gamma
        )

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()
        self.contras_loss = ContrastiveLoss(temperature=0.7)
        self.triplet_loss = HardTripletLoss(margin=0.5)
        self.compressed_param = {}
        self.energy = None
        self.gamma = args.gamma
        self.lamda_ = args.lamda
        self.use_nkd_loss = args.use_nkd_loss # using NDK Loss
        self.use_ct_loss  = args.use_ct_loss # using Contrastive loss
        self.use_dsvd = args.use_dsvd # Using SVD for factorize weighted matrix
        print(self.use_nkd_loss, self.use_ct_loss, self.use_dsvd, self.gamma, self.lamda_)  

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            loss_e = 0
            loss_g_e = 0
            loss_h  = 0
            loss_g_h = 0
            loss_nkd_e = 0
            loss_ct_e = 0
            
            # random choice 2 batch data in loader

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)
                #normalized KD
                N, c = output.shape
                s_i = F.log_softmax(output)
                t_i = F.softmax(output_g, dim=1)
                # random choice 2 sample in batch
                # data_add_noise = x + torch.randn_like(x) * 0.1
                # l_const = lipschitz_constant_v2(self.model, x, data_add_noise)

                if len(y.size()) > 1:
                    label = torch.max(y, dim=1, keepdim=True)[1]
                else:
                    label = y.view(len(y), 1)
                # print("S_i: ",s_i.shape)
                # print("T_i: ",t_i.shape)
                # print("label: ", y.shape)


                s_t = torch.gather(s_i, 1, label)
                t_t = torch.gather(t_i, 1, label)
                loss_t = -(t_t * s_t).mean()

                mask = torch.ones_like(output).scatter_(1, label, 0).bool()
                logit_s = output[mask].reshape(N, -1) # set local as student
                logit_t = output_g[mask].reshape(N, -1) # set global as teacher

                # N*class
                S_i = F.log_softmax(logit_s/1)
                T_i = F.softmax(logit_t/1, dim=1)

                loss_non =  (T_i * S_i).sum(dim=1).mean()
                loss_non = - self.lamda_ * (self.gamma**2) * loss_non

                loss_nkd = loss_t + loss_non

                #contrastive loss
                loss_ct = self.contras_loss(rep, rep_g)

                #triplet loss
                # tpl_local = self.triplet_loss(rep, label)
                # tpl_global = self.triplet_loss(rep_g, label)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)





                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / ( CE_loss + CE_loss_g)
                L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) / (CE_loss + CE_loss_g)



                # L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                # check all use loss: 
                if self.use_nkd_loss and self.use_ct_loss:
                    loss = CE_loss + L_d + L_d_g + loss_nkd + loss_ct
                    loss_g = CE_loss_g + L_d + L_d_g + loss_nkd + loss_ct
                    loss_nkd_e += loss_nkd.item()
                    loss_ct_e += loss_ct.item()
                elif self.use_nkd_loss and not self.use_ct_loss:
                    loss = CE_loss + L_d + L_d_g + loss_nkd
                    loss_g = CE_loss_g + L_d + L_d_g + loss_nkd
                    loss_nkd_e += loss_nkd.item()
                    loss_ct_e += 0
                elif not self.use_nkd_loss and self.use_ct_loss:
                    loss = CE_loss + L_d + L_d_g + loss_ct
                    loss_g = CE_loss_g + L_d + L_d_g + loss_ct
                    loss_nkd_e += 0
                    loss_ct_e += loss_ct.item()
                else:
                    loss = CE_loss + L_d + L_d_g
                    loss_g = CE_loss_g + L_d + L_d_g
                    loss_nkd_e += 0
                    loss_ct_e += 0
                
                # l_const_g = lipschitz_constant_v2(self.global_model, x1, x2)
                # l_const = lipschitz_constant(self.model, x1)


                self.optimizer.zero_grad()
                self.optimizer_g.zero_grad()
                self.optimizer_W.zero_grad()
                loss.backward(retain_graph=True)
                loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.W_h.parameters(), 10)
                self.optimizer.step()
                self.optimizer_g.step()
                self.optimizer_W.step()
                loss_e += loss.item()
                loss_g_e += loss_g.item()
                loss_h +=  L_d.item()
                loss_g_h += L_d_g.item()
                # loss_nkd_e += loss_nkd.item()
                # loss_ct_e += loss_ct.item()
                # l_rep_e += L_h.item()
                # l_const_g_e += l_const_g.item()
            print(loss_nkd_e, loss_ct_e)
            print(f"\033[94mEpoch: {epoch}|  NKD Loss: {round(loss_nkd_e/len(trainloader), 4)} | CT Loss: {round(loss_ct_e/len(trainloader), 4)}| \033[0m")
            print(f"\033[91mEpoch: {epoch}|  Loss:  {round(loss_e/len(trainloader), 4)} |Global loss: {round(loss_g_e/len(trainloader), 4)}| Local H loss: {round(loss_h/len(trainloader), 4)}  | Global H loss: {round(loss_g_h/len(trainloader), 4)} \033[0m")
            # print(f"\033[94m Lipschitz constant: {l_const_e/len(trainloader)} | Lipschitz constant global:  {l_const_g_e/len(trainloader)} \033[0m")

        # self.model.cpu()

        self.decomposition()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()
            self.learning_rate_scheduler_W.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_parameters(self, global_param, energy):
        # recover
        for k in global_param.keys():
            if len(global_param[k]) == 3:
                # use np.matmul to support high-dimensional CNN param
                global_param[k] = np.matmul(global_param[k][0] * global_param[k][1][..., None, :], global_param[k][2])

        for name, old_param in self.global_model.named_parameters():
            if name in global_param:
                old_param.data = torch.tensor(global_param[name], device=self.device).data.clone()
        self.energy = energy

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)

                #normalized KD
                N, c = output.shape
                s_i = F.log_softmax(output)
                t_i = F.softmax(output_g, dim=1)

                if len(y.size()) > 1:
                    label = torch.max(y, dim=1, keepdim=True)[1]
                else:
                    label = y.view(len(y), 1)
                # print("S_i: ",s_i.shape)
                # print("T_i: ",t_i.shape)
                # print("label: ", y.shape)


                s_t = torch.gather(s_i, 1, label)
                t_t = torch.gather(t_i, 1, label)
                loss_t = -(t_t * s_t).mean()

                mask = torch.ones_like(output).scatter_(1, label, 0).bool()
                logit_s = output[mask].reshape(N, -1) # set local as student
                logit_t = output_g[mask].reshape(N, -1) # set global as teacher

                # N*class
                S_i = F.log_softmax(logit_s/1)
                T_i = F.softmax(logit_t/1, dim=1)

                loss_non =  (T_i * S_i).sum(dim=1).mean()
                loss_non = - 1.5 * (1**2) * loss_non

                loss_nkd = loss_t + loss_non
                loss_ct = self.contras_loss(rep, rep_g)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                loss = CE_loss + L_d + L_h + loss_nkd + loss_ct
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    def decomposition(self):
        self.compressed_param = {}
        for name, param in self.global_model.named_parameters():
            param_cpu = param.detach().cpu().numpy()
            # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
            if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (2, 0, 1))
                    v = np.transpose(v, (2, 3, 0, 1))
                threshold=0
                if np.sum(np.square(sigma))==0:
                    compressed_param_cpu=param_cpu
                else:
                    for singular_value_num in range(len(sigma)):
                        if np.sum(np.square(sigma[:singular_value_num]))>self.energy*np.sum(np.square(sigma)):
                            threshold=singular_value_num
                            break
                    u=u[:, :threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold, :]
                    # support high-dimensional CNN param
                    if len(u.shape)==4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))
                    compressed_param_cpu=[u,sigma,v]
            elif 'embeddings' not in name:
                compressed_param_cpu=param_cpu

            self.compressed_param[name] = compressed_param_cpu 