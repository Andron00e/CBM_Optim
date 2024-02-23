# add configure_optimizers and bs_muls function
# add .test method and test_loader if not None
# test evaluate load instead of load_metric from datasets
# rename preprocessed loader to loader
# draw norm hists and compute norm diffs functions to be released

import os
import time
import warnings
from graph_plot_tools import *
from utils import *
from configs import *
from evaluate import load
from IPython import display
from matplotlib import animation

def init_history(run_name, nets, opts, opt_names, bs_muls):
    hist = []
    for (net, optimizer, opt_name, bs_mul) in zip(nets, opts, opt_names, bs_muls):
        net.to(device)
        hist.append({
            "run_name": run_name,
            "name": opt_name,
            "bs_mul": bs_mul,
            "train_loss": [], "train_x": [],
            "val_loss": [], "val_x": [],
            "train_cbl_loss": [], "val_cbl_loss": [],
            "train_acc_top_1": [], "train_acc_top_5": [],
            "test_acc_top_1": [], "test_acc_top_5": [],
            "val_acc_top_1": [], "val_acc_top_5": [],
            "val_precision": [], "val_recall": [],
            "val_f1": [], "test_precision": [],
            "test_recall": [], "test_f1": [],
            "norm_diffs": [],
            "epochs_x": [],
            "norm_diffs_x": [],
            "total_steps": 0,
            "prev_val_eval_step": 0,
            "prev_grad_norms_eval_step": 0,
            "batch_end": True
        })
        return hist
    

class BottleneckTrainer:
    def __init__(self, 
                 nets, 
                 opts, 
                 hist, 
                 device, 
                 train_loader, 
                 val_loader, 
                 test_loader,
                 training_method, 
                 run_name, 
                 num_epochs=10, 
                 lr_decay=1.,
                 batch_mul_step_count=500,
                 norm_diffs_step_count=500,
                 val_step_count=500,
                ):
        self.nets = nets
        self.opts = opts
        self.hist = hist
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = torch.nn.CrossEntropyLoss()
        self.training_method = training_method
        self.criterion_gumbel = criterion_gumbel
        self.criterion_cbl = criterion_cbl
        self.criterion_l1 = criterion_l1
        self.precision_metric = load("precision")
        self.recall_metric = load("recall")
        self.f1_metric = load("f1")
        self.lr_decay = lr_decay
        self.run_name = run_name
        self.num_epochs = 10
        self.batch_mul_step_count = batch_mul_step_count
        self.norm_diffs_step_count = norm_diffs_step_count
        self.val_step_count = val_step_count
        self.calc_norm_diffs = False

    def _update_learning_rate(self, param_groups, decay):
        for g in param_groups:
            g['lr'] = g['lr'] * decay        

    def train(self):
        num_epochs = self.num_epochs
        batch_mul_step_count = self.batch_mul_step_count
        norm_diffs_step_count = self.norm_diffs_step_count
        val_step_count = self.val_step_count
        calc_norm_diffs = self.calc_norm_diffs

        run_name = self.run_name
        os.makedirs(run_name, exist_ok=True)

        for epoch in range(num_epochs):
            for net, opt, net_hist in zip(self.nets, self.opts, self.hist):
                net.to(self.device)
                bs_mul = net_hist["bs_mul"]
                total_steps = net_hist["total_steps"]

                if net_hist["bs_mul"] == "linear":
                    bs_mul = int(total_steps / batch_mul_step_count) + 1

                net_hist["epochs_x"].append(total_steps)
                optimizer_cbl, optimizer_head = opt

                for i, batch in enumerate(self.train_loader, 0):

                    if calc_norm_diffs and ((total_steps - net_hist["prev_grad_norms_eval_step"]) > norm_diffs_step_count or
                                            len(net_hist["norm_diffs"]) == 0) and net_hist["batch_end"]:
                        net_hist["prev_grad_norms_eval_step"] = total_steps
                        net.eval()

                        norm_diffs = self._compute_norm_diffs(net, optimizer_head, None, train_loader_wo_crops, train_loader_wo_crops, repeats=5)

                        net_hist["norm_diffs"].append(norm_diffs)
                        net_hist["norm_diffs_x"].append(total_steps)
                        net.train()

                    net_hist["batch_end"] = False

                    self._update_learning_rate(optimizer_head.param_groups, self.lr_decay)

                    inputs, labels = batch
                    inputs, targets = inputs.to(self.device), torch.LongTensor(labels).to(self.device)
                    cbl_logits, logits = net(**inputs)
                    
                    if self.training_method == "gumbel":
                        cbl_loss = self.criterion_gumbel(cbl_logits)
                    elif self.training_method == "contrastive":
                        cbl_loss = self.criterion_cbl(cbl_logits)
                    elif self.training_method == "l1":
                        cbl_loss = self.criterion_l1(net) / cbl_logits.squeeze().shape[1]
                            
                    cbl_loss.backward(retain_graph=True)
                    ce_loss = self.criterion(logits, targets) / bs_mul
                    ce_loss.backward()

                    if total_steps % bs_mul == bs_mul - 1:
                        optimizer_cbl.step()
                        optimizer_cbl.zero_grad()
                        optimizer_head.step()
                        optimizer_head.zero_grad()
                        net_hist["batch_end"] = True

                    net_hist["train_loss"].append(ce_loss.detach().cpu().item() * bs_mul)
                    net_hist["train_cbl_loss"].append(cbl_loss.detach().cpu().item() * bs_mul)
                    net_hist["train_x"].append(total_steps)

                    if total_steps % bs_mul == bs_mul - 1:
                        if net_hist["bs_mul"] == "linear":
                            net_hist["bs_mul"] = int(int(total_steps) / batch_mul_step_count) + 1
                            bs_mul = net_hist["bs_mul"]

                    top_1, top_5 = self._accuracy(logits, targets, topk=(1, 5))
                    net_hist["train_acc_top_1"].append(top_1.detach().cpu().item())
                    net_hist["train_acc_top_5"].append(top_5.detach().cpu().item())

                    prev_val_eval_step = net_hist["prev_val_eval_step"]
                    if (total_steps - prev_val_eval_step) > val_step_count and net_hist["batch_end"]:
                        net_hist["prev_val_eval_step"] = total_steps

                        net.eval()

                        val_cbl_losses, val_ce_losses, \
                        val_top_1_accs, val_top_5_accs, \
                        val_top_1_precisions, val_top_1_recalls, \
                        val_top_1_f1scores = self._evaluate(net, self.val_loader)

                        net_hist["val_loss"].append(np.mean(val_ce_losses))
                        net_hist["val_cbl_loss"].append(np.mean(val_cbl_losses))

                        net_hist["val_acc_top_1"].append(np.mean(val_top_1_accs))
                        net_hist["val_acc_top_5"].append(np.mean(val_top_5_accs))

                        net_hist["val_precision"].append(np.mean(val_top_1_precisions))
                        net_hist["val_recall"].append(np.mean(val_top_1_recalls))
                        net_hist["val_f1"].append(np.mean(val_top_1_f1scores))

                        net_hist["val_x"].append(total_steps)
                        net.train()

                    if total_steps % 10 == 0:
                        start_time = time.time()
                        self._display_results(net_hist)

                    total_steps += 1

                    net_hist["total_steps"] = total_steps

                self.save_checkpoint(epoch, net, optimizer_cbl, optimizer_head, net_hist)
            
        print('Finished Training')

    def save_checkpoint(self, epoch, net, optimizer_cbl, optimizer_head, net_hits):
        state_dict = {
                    "epoch": epoch,
                    str(net_hist["val_acc_top_1"][-1])[:5]+'_model_state_dict': net.state_dict(),
                    "optimizer_cbl_state_dict": optimizer_cbl.state_dict(),
                    "optimizer_head_state_dict": optimizer_head.state_dict(),
                    "loss_train": net_hist["train_loss"],
                    "cbl_loss_train": net_hist["train_cbl_loss"],
                    "loss_val": net_hist["val_loss"],
                    "cbl_loss_val": net_hist["val_cbl_loss"],
                }
        torch.save(state_dict, 
                   os.path.join(self.run_name, str(net_hist["val_acc_top_1"][-1])[:5]+f'_checkpoint_{epoch}_epoch.pth'))
    
    def _compute_norm_diffs(self, net, optimizer, scheduler, train_loader, valid_loader, repeats):
        pass

    def _accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    @torch.no_grad()
    def _evaluate(self, net, loader):
        cbl_losses, ce_losses = [], []
        top_1_accs, top_5_accs = [], []
        top_1_precisions, top_1_recalls = [], []
        top_1_f1scores = []
        
        with torch.no_grad():
            for step, batch in enumerate(loader, 0):
                warnings.filterwarnings("ignore")
                inputs, labels = batch
                inputs, targets = inputs.to(self.device), torch.LongTensor(labels).to(self.device)
                cbl_logits, logits = net(**inputs)

                if self.training_method == "gumbel":
                    cbl_loss = self.criterion_gumbel(cbl_logits)
                elif self.training_method == "contrastive":
                    cbl_loss = self.criterion_cbl(cbl_logits)
                elif self.training_method == "l1":
                    cbl_loss = self.criterion_l1(net) / cbl_logits.squeeze().shape[1]
                
                cbl_losses.append(cbl_loss.detach().cpu().item())

                ce_loss = self.criterion(logits, targets)
                ce_losses.append(ce_loss.detach().cpu().item())

                top_1, top_5 = self._accuracy(logits, targets, topk=(1, 5))
                top_1_accs.append(top_1.detach().cpu().item())
                top_5_accs.append(top_5.detach().cpu().item())

                precs = self.precision_metric.compute(predictions=logits.argmax(dim=-1).cpu(), 
                                                 references=targets.cpu(), 
                                                 average='weighted')
                recs = self.recall_metric.compute(predictions=logits.argmax(dim=-1).cpu(), 
                                             references=targets.cpu(), 
                                             average='weighted')
                f1 = self.f1_metric.compute(predictions=logits.argmax(dim=-1).cpu(), 
                                       references=targets.cpu(), 
                                       average='weighted', 
                                       labels=np.unique(logits.argmax(dim=-1).cpu()))

                top_1_precisions.append(precs['precision'])
                top_1_recalls.append(recs['recall'])
                top_1_f1scores.append(f1['f1'])
                
        return cbl_losses, ce_losses, top_1_accs, top_5_accs, top_1_precisions, top_1_recalls, top_1_f1scores
    
    @torch.no_grad()
    def test(self):
        if self.test_loader is None:
            print("No test loader is provided!", "\n")
        else:
            print("Begin Testing")
            for net, opt, net_hist in tqdm(zip(self.nets, self.opts, self.hist)):
                net.to(self.device)
                net.eval()
                test_cbl_losses, test_ce_losses, \
                test_top_1_accs, test_top_5_accs, \
                test_top_1_precisions, test_top_1_recalls, \
                test_top_1_f1scores = self._evaluate(net, self.test_loader)

                net_hist["test_acc_top_1"].append(np.mean(test_top_1_accs))
                net_hist["test_acc_top_5"].append(np.mean(test_top_5_accs))

                net_hist["test_precision"].append(np.mean(test_top_1_precisions))
                net_hist["test_recall"].append(np.mean(test_top_1_recalls))
                net_hist["test_f1"].append(np.mean(test_top_1_f1scores))

                torch.save({
                    str(net_hist["test_acc_top_1"][-1])[:5]+'_model.bin': net.state_dict
                    }, os.path.join(self.run_name, str(net_hist["test_acc_top_1"][-1])[:5]+'_model.bin')
                )
                print(net_hist["test_acc_top_1"][-1], "\n")
            print('Finished Testing')
            
    
    def _display_results(self, net_hist, wait=True, clear_output=True):
        if clear_output:
            display.clear_output(wait=wait)
        
        grouped_hist = group_uniques_full(self.hist, ["train_loss", "val_loss", "val_acc_top_1", "train_acc_top_1"])
        
        fig = plt.figure(figsize=(15, 8 + 2 * ((len(grouped_hist) + 2) // 3)))
        gs = GridSpec(4 + 2 * ((len(grouped_hist) + 2) // 3), 3, figure=fig, wspace=0.3, hspace=0.9)
        ax1 = fig.add_subplot(gs[0:4,:2])
        ax2 = fig.add_subplot(gs[0:2,2])
        ax3 = fig.add_subplot(gs[2:4,2])
        make_loss_plot(ax1, grouped_hist, loss_name="CE Loss", eps=0.01, make_val=True, alpha=0.9)
        make_accuracy_plot(ax2, grouped_hist, eps=0.01, make_train=True, make_val=False, top_k=1, alpha=0.9)
        make_accuracy_plot(ax3, grouped_hist, eps=0.01, make_train=False, make_val=True, top_k=1, alpha=0.9)
        plt.draw()
        
        def animate(frame):
            row = frame // 3
            col = frame % 3
            if col == 0:
                ax1 = make_loss_plot(ax1, grouped_hist, loss_name="CE Loss", eps=0.01, make_val=True, alpha=0.9)
            elif col == 1:
                ax2 = make_accuracy_plot(ax2, grouped_hist, eps=0.01, make_train=True, make_val=False, top_k=1, alpha=0.9)
            elif col == 2:
                ax3 = make_accuracy_plot(ax3, grouped_hist, eps=0.01, make_train=False, make_val=True, top_k=1, alpha=0.9)
            
            plt.draw()

        frames = len(grouped_hist)*3
        anim = animation.FuncAnimation(plt.gcf(), animate, frames=frames, interval=1000, blit=False)
        plt.show(block=True)
