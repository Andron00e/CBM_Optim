import os
import torch
import ast
import numpy as np
from copy import deepcopy
import json
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython import display
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import bisect
import scipy
from scipy import stats

device = torch.device("cpu")


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = len(target)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def smooth(a, eps=0.01):
    b = [a[0]]
    for e in a[1:]:
        b.append(b[-1] * (1 - eps) + e * eps)
    return b


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def group_uniques(hist, one_optim_hist, group_unique, loss_name):
    if not group_unique:
        return one_optim_hist[loss_name]
    else:
        res = [0] * len(one_optim_hist[loss_name])
        unique_name = one_optim_hist["name"]
        k = 0
        for one_optim_hist in hist:
            if one_optim_hist["name"] == unique_name:
                k += 1
                for i, elem in enumerate(one_optim_hist[loss_name]):
                    res[i] += elem
        res = [elem / k for elem in res]
        return res


def make_loss_plot(
    ax,
    hist,
    loss_name,
    eps=0.01,
    alpha=0.5,
    make_train=True,
    make_val=True,
    starting_epoch=0,
    group_unique=False,
):
    if len(hist) < 7:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 0.9, len(hist)))
    unique_tried = {}
    for one_optim_hist in hist:
        unique_tried[one_optim_hist["name"]] = False

    for i, one_optim_hist in enumerate(hist):
        label = one_optim_hist["name"]

        if group_unique and unique_tried[label]:
            continue
        unique_tried[label] = True

        if len(one_optim_hist["train_x"]) == 0:
            continue

        epochs_x = one_optim_hist["epochs_x"][starting_epoch:]
        start = int(epochs_x[0])

        if make_train:
            train_y = group_uniques(hist, one_optim_hist, group_unique, "train_loss")
            smoothed_train_y = smooth(train_y, eps=eps)[start:]
            train_x = one_optim_hist["train_x"][start:]
            ax.plot(
                train_x,
                smoothed_train_y,
                label=label + " (train)",
                alpha=alpha,
                color=colors[i],
                linestyle="-",
            )

        if make_val and len(one_optim_hist["val_x"]) > 0:
            val_y = group_uniques(hist, one_optim_hist, group_unique, "val_loss")
            val_x = one_optim_hist["val_x"]
            ind = bisect.bisect_left(val_x, start)
            ax.plot(
                val_x[ind:],
                val_y[ind:],
                label=label + " (val)",
                alpha=alpha,
                color=colors[i],
                linestyle="--",
            )

        # epoch sep lines
        for x in epochs_x:
            ax.axvline(x, linestyle="--", color=colors[i], alpha=0.2)

    if make_train and make_val:
        ax.set_title("{} on train/val".format(loss_name))
    elif make_train:
        ax.set_title("{} on train".format(loss_name))
    elif make_val:
        ax.set_title("{} on val".format(loss_name))
    ax.set_ylabel("{}".format(loss_name))
    ax.set_xlabel("iteration")
    ax.grid(True)
    ax.legend()

    return ax


def make_metrics_plot(
    ax,
    hist,
    eps=0.01,
    alpha=0.5,
    make_train=True,
    make_val=True,
    starting_epoch=0,
    metric_name="acc_top_1",
    title="top-1 accuracy",
):
    if len(hist) < 7:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 0.9, len(hist)))

    for i, one_optim_hist in enumerate(hist):
        label = one_optim_hist["name"]

        if len(one_optim_hist["train_x"]) == 0:
            continue

        epochs_x = one_optim_hist["epochs_x"][starting_epoch:]
        start = int(epochs_x[0])

        if make_train:
            train_acc_top = one_optim_hist["train_{}".format(metric_name)]
            smoothed_train_acc_top = smooth(train_acc_top, eps=eps)[start:]
            train_x = one_optim_hist["train_x"][start:]
            ax.plot(
                train_x,
                smoothed_train_acc_top,
                label=label + " (train)",
                alpha=alpha,
                color=colors[i],
                linestyle="-",
            )

        if make_val and len(one_optim_hist["val_x"]) > 0:
            val_acc_top = one_optim_hist["val_{}".format(metric_name)]
            val_x = one_optim_hist["val_x"]
            ind = bisect.bisect_left(val_x, start)
            ax.plot(
                val_x[ind:],
                val_acc_top[ind:],
                label=label + " (val)",
                alpha=alpha,
                color=colors[i],
                linestyle="--",
            )

        # epoch sep lines
        for x in epochs_x:
            ax.axvline(x, linestyle="--", color=colors[i], alpha=0.2)

    if make_train and make_val:
        ax.set_title("{} on train/val".format(title))
    elif make_train:
        ax.set_title("{} on train".format(title))
    elif make_val:
        ax.set_title("{} on val".format(title))

    ax.set_ylabel("{}".format(title))
    ax.set_xlabel("Iteration")
    ax.grid(True)

    return ax


def make_accuracy_plot(
    ax,
    hist,
    eps=0.01,
    alpha=0.5,
    top_k=1,
    make_train=True,
    make_val=True,
    starting_epoch=0,
):
    return make_metrics_plot(
        ax,
        hist,
        eps,
        alpha,
        make_train,
        make_val,
        starting_epoch,
        metric_name="acc_top_{}".format(top_k),
        title="top-{} accuracy".format(top_k),
    )


def make_plot(
    ax,
    hist,
    y_name,
    x_name,
    title="Training",
    y_label="loss",
    x_label="iteration",
    eps=0.01,
    alpha=0.5,
    starting_epoch=0,
    draw_epoch_lines=True,
    grid=True,
    legend=True,
    mark_every_count=15,
    title_fontsize=14,
    xy_label_fontsize=10,
    line_params={},
):
    if len(hist) < 7:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 0.9, len(hist)))

    for i, one_optim_hist in enumerate(hist):
        label = one_optim_hist["name"]
        if label in line_params:
            line_style = line_params[label]["line_style"]
            line_color = line_params[label]["line_color"]
            line_marker = line_params[label]["line_marker"]
            mark_every = line_params[label]["mark_every"]
            label = line_params[label]["name"]
        else:
            line_style = "-"
            line_color = colors[i]
            line_marker = None
            mark_every = None

        if len(one_optim_hist["epochs_x"]) <= starting_epoch:
            continue

        epochs_x = one_optim_hist["epochs_x"][starting_epoch:]
        start = int(epochs_x[0])

        if len(one_optim_hist[y_name]) == 0:
            continue

        smoothed_y = smooth(one_optim_hist[y_name], eps=eps)
        x = one_optim_hist[x_name]

        ind = bisect.bisect_left(x, start)
        smoothed_y = smoothed_y[ind:]
        x = x[ind:]

        if mark_every == "auto":
            mark_every = len(x) // (mark_every_count + 1)
            mark_every = np.arange((i * mark_every) // (len(hist)), len(x), mark_every)

        ax.plot(
            x,
            smoothed_y,
            label=label,
            alpha=alpha,
            color=line_color,
            linestyle=line_style,
            marker=line_marker,
            markevery=mark_every,
        )

        # epoch sep lines
        if draw_epoch_lines:
            for x in epochs_x:
                ax.axvline(x, linestyle="--", color=line_color, alpha=0.2)

    plt.rc("font", size=xy_label_fontsize)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=xy_label_fontsize)
    ax.set_xlabel(x_label, fontsize=xy_label_fontsize)
    ax.grid(grid)
    if legend:
        ax.legend()

    return ax


def draw_norm_hist(ax, norm_diffs, bins_n=100, draw_normal=True):
    counts, bins = np.histogram(norm_diffs, bins_n, density=True)

    if draw_normal:
        mu = np.mean(norm_diffs)
        sigma = np.sqrt(np.mean((norm_diffs - mu) ** 2))
        temp2 = np.linspace(bins[0], bins[-1], bins_n)
        y = scipy.stats.norm.pdf(temp2, mu, sigma)
        temp2 = temp2[y > (counts[counts > 0]).min()]

        ax.plot(temp2, scipy.stats.norm.pdf(temp2, mu, sigma), linewidth=2, color="red")

    ax.hist(bins[:-1], bins, weights=counts)

    return ax


def draw_norm_hists_for_different_models(
    fig, subplotspec_outer, hist, bins_n=100, draw_normal=True, number=-1
):
    h = (len(hist) + 2) // 3
    sgs = subplotspec_outer.subgridspec(h, 3, wspace=0.15, hspace=0.25)
    for i, one_optim_hist in enumerate(hist):
        ax = fig.add_subplot(sgs[i // 3, i % 3])
        if len(one_optim_hist["norm_diffs"]) > 0:
            ax = draw_norm_hist(
                ax,
                one_optim_hist["norm_diffs"][number],
                bins_n=bins_n,
                draw_normal=draw_normal,
            )
            label = one_optim_hist["name"]
            ax.set_title(
                "{},\n batch_count={}, skew={:0.2f}".format(
                    label,
                    len(one_optim_hist["norm_diffs"][number]),
                    stats.skew(one_optim_hist["norm_diffs"][number]),
                ),
                fontsize=11,
            )
        ax.set_ylabel("Density")
        ax.set_xlabel("Noise norm")
        ax.grid(True)

    ax = plt.Subplot(fig, subplotspec_outer)
    ax.set_title(
        "Distribution of the gradient noise for different optimizers",
        y=1.15,
        fontsize=14,
    )
    ax.set_frame_on(False)
    ax.axis("off")
    fig.add_subplot(ax)

    return ax


def draw_norm_hists_for_one_model(
    ax,
    hist,
    title,
    y_label="density",
    x_label="noise norm",
    bins_n=100,
    draw_normal=True,
    number=-1,
    grid=True,
    add_batch_count=True,
    round_batch_count=True,
    title_fontsize=14,
    xy_label_fontsize=10,
):
    one_optim_hist = hist[0]
    ax = draw_norm_hist(
        ax, one_optim_hist["norm_diffs"][number], bins_n=bins_n, draw_normal=draw_normal
    )

    batch_count = len(one_optim_hist["norm_diffs"][number])
    if round_batch_count:
        batch_count = "~{}k".format(batch_count // 1000)
    if add_batch_count:
        title = "{}, batch count={}".format(title, batch_count)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=xy_label_fontsize)
    ax.set_xlabel(x_label, fontsize=xy_label_fontsize)
    ax.grid(grid)

    return ax


def draw_norm_hists_evolution(
    fig, subplotspec_outer, one_optim_hist, bins_n=100, draw_normal=True, w=None
):
    if w is None:
        w = len(one_optim_hist["norm_diffs"])

    label = one_optim_hist["name"]

    h = (len(one_optim_hist["norm_diffs"]) + w - 1) // w
    sgs = subplotspec_outer.subgridspec(h, w, wspace=0.15, hspace=0.05)
    for i, norm_diffs in enumerate(one_optim_hist["norm_diffs"]):
        ax = fig.add_subplot(sgs[i // w, i % w])
        if len(norm_diffs) > 0:
            ax = draw_norm_hist(ax, norm_diffs, bins_n=bins_n, draw_normal=draw_normal)
            step = 500 * i
            if "norm_diffs_x" in one_optim_hist:
                step = one_optim_hist["norm_diffs_x"][i]
            ax.set_title(
                "step = {},\n batch_count={}, skew={:0.2f}".format(
                    step, len(norm_diffs), stats.skew(norm_diffs)
                ),
                fontsize=11,
            )
        ax.set_ylabel("density")
        ax.set_xlabel("noise norm")
        ax.grid(True)

    ax = plt.Subplot(fig, subplotspec_outer)
    ax.set_title(
        "Evolution of the gradient noise for {}".format(label), y=1.15, fontsize=14
    )
    ax.set_frame_on(False)
    ax.axis("off")
    fig.add_subplot(ax)

    return ax


from copy import deepcopy


def recursive_to(param, device):
    if isinstance(param, torch.Tensor):
        param.data = param.data.to(device)
        if param._grad is not None:
            param._grad.data = param._grad.data.to(device)
    elif isinstance(param, dict):
        for subparam in param.values():
            recursive_to(subparam, device)
    elif isinstance(param, list):
        for subparam in param:
            recursive_to(subparam, device)


def optimizer_to(optim, device):
    for param_group in optim.param_groups:
        for param in param_group.values():
            recursive_to(param, device)


def group_uniques_full(hist, losses_to_average, verbose=False, group_norm_diffs=False):
    grouped_hist = {}
    unique_tried = {}
    for one_optim_hist in hist:
        unique_tried[one_optim_hist["name"]] = False

    for one_optim_hist in hist:
        label = one_optim_hist["name"]
        if not unique_tried[label]:
            unique_tried[label] = True
            grouped_hist[label] = {
                "hist": deepcopy(one_optim_hist),
                "repeats": {
                    loss_name: [1] * len(one_optim_hist[loss_name])
                    for loss_name in losses_to_average
                },
            }
            if group_norm_diffs:
                grouped_hist[label]["hist"]["norm_diffs"] = [
                    [np.array(x)] for x in grouped_hist[label]["hist"]["norm_diffs"]
                ]
            continue

        for loss_name in losses_to_average:
            losses = one_optim_hist[loss_name]

            for i, loss_elem in enumerate(losses):
                if i < len(grouped_hist[label]["hist"][loss_name]):
                    grouped_hist[label]["hist"][loss_name][i] += loss_elem
                else:
                    grouped_hist[label]["hist"][loss_name].append(loss_elem)

                if i >= len(grouped_hist[label]["repeats"][loss_name]):
                    grouped_hist[label]["repeats"][loss_name].append(0)
                grouped_hist[label]["repeats"][loss_name][i] += 1

        if group_norm_diffs:
            if len(grouped_hist[label]["hist"]["norm_diffs"]) == 0:
                if "norm_diffs_x" in one_optim_hist:
                    grouped_hist[label]["hist"]["norm_diffs_x"] = one_optim_hist[
                        "norm_diffs_x"
                    ]
                grouped_hist[label]["hist"]["norm_diffs"] = [
                    [np.array(x)] for x in one_optim_hist["norm_diffs"]
                ]
            else:
                for x, y in zip(
                    grouped_hist[label]["hist"]["norm_diffs"],
                    one_optim_hist["norm_diffs"],
                ):
                    x.append(np.array(y))

    for key in grouped_hist:
        one_optim_hist = grouped_hist[key]
        if verbose and len(one_optim_hist["repeats"][losses_to_average[0]]) > 0:
            repeats_1 = float(one_optim_hist["repeats"][losses_to_average[0]][0])
            print(
                "Repeats_1 = {}, Name = {}".format(
                    repeats_1, one_optim_hist["hist"]["name"]
                )
            )
        for loss_name in losses_to_average:
            for i in range(len(one_optim_hist["hist"][loss_name])):
                repeats = one_optim_hist["repeats"][loss_name][i]
                one_optim_hist["hist"][loss_name][i] /= repeats

        if group_norm_diffs:
            for i, group in enumerate(one_optim_hist["hist"]["norm_diffs"]):
                means = []
                stds = []
                for j, elem in enumerate(group):
                    means.append(elem.mean())
                    stds.append(elem.std())
                    group[j] = (elem - elem.mean()) / elem.std()
                mean = np.mean(means)
                std = np.mean(stds)
                one_optim_hist["hist"]["norm_diffs"][i] = (
                    np.concatenate(group) * std + mean
                )

    grouped_hist = [grouped_hist[x]["hist"] for x in grouped_hist]

    return grouped_hist


def load_hist_jsons(hists_names_list, path="./models"):
    hists = []
    for hist_name in hists_names_list:
        with open(r"{}/{}.json".format(path, hist_name), "r") as read_file:
            hist = json.load(read_file)
            hists += hist
    return hists


def rec_hist_from_json(h, key):
    for i in range(len(h)):
        if key == "val_norm_diffs":
            h[i] = ast.literal_eval(h[i])
            for j in range(len(h[i])):
                h[i][j] = float(h[i][j])
        elif isinstance(h[i], list):
            rec_hist_from_json(h[i], key)
        else:
            h[i] = float(h[i])


def hist_from_json(hists):
    for h in hists:
        for key in h:
            if isinstance(h[key], list):
                rec_hist_from_json(h[key], key)
    return hists


import scipy
from scipy import stats
from tqdm.notebook import tqdm


def get_batch_grad(model):
    gr = []
    for i in model.parameters():
        if i.requires_grad:
            gr.append(i.grad.view(-1))
    return torch.cat(gr)


def get_loss(model, criterion, batch):
    inputs, labels = batch
    inputs, targets = inputs.to(device), torch.LongTensor(labels).to(device)
    cbl_logits, logits = model(**inputs)
    loss = criterion(logits, targets)
    return loss


def compute_full_grad(model, optimizer, criterion, dataloader_for_full_grad):
    fully_grad = []
    optimizer.zero_grad()

    print("Computing full gradient")
    with tqdm(total=len(dataloader_for_full_grad)) as pbar:
        for step, batch in enumerate(dataloader_for_full_grad):
            loss = get_loss(model, criterion, batch)
            loss.backward()

            if fully_grad != []:
                fully_grad = (
                    fully_grad
                    + get_batch_grad(model) * dataloader_for_full_grad.batch_size
                )
            else:
                fully_grad = get_batch_grad(model) * dataloader_for_full_grad.batch_size
            optimizer.zero_grad()

            pbar.update(1)

    return fully_grad / (step * dataloader_for_full_grad.batch_size)


def compute_norm_diffs(
    model,
    optimizer,
    criterion,
    dataloader_for_full_grad,
    dataloader,
    full_grad=None,
    repeats=1,
):
    if full_grad is None:
        full_grad = compute_full_grad(
            model, optimizer, criterion, dataloader_for_full_grad
        )
    mini_norms = []
    optimizer.zero_grad()

    print("Computing norm diffs")
    with tqdm(total=repeats * len(dataloader)) as pbar:
        for _ in range(repeats):
            for step, batch in enumerate(dataloader):
                loss = get_loss(model, criterion, batch)
                loss.backward()

                mini_norms.append((get_batch_grad(model) - full_grad).norm().item())
                optimizer.zero_grad()

                pbar.update(1)

    return np.array(mini_norms)


""" GENERAL """
batch_mul_step_count = 5000
norm_diffs_step_count = 5000
val_step_count = 5000
calc_norm_diffs = False


class BottleneckTrainer:
    def __init__(
        self,
        nets,
        opts,
        num_epochs,
        criterion_cbl,
        criterion_head,
        train_loader,
        val_loader,
        test_loader,
        task_name,
        opt_names,
        bs_muls,
        lr_decay=1,
    ):
        self.nets = nets
        self.opts = opts
        self.criterion_cbl = criterion_cbl
        self.criterion_head = criterion_head
        self.num_epochs = num_epochs
        self.task_name = task_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.train_loader_for_full_grad = train_loader_for_full_grad
        self.opt_names = opt_names
        self.bs_muls = bs_muls
        self.lr_decay = lr_decay
        self.hist = self.initialize_history()

    def initialize_history(self):
        hist = []
        for net, optimizer, opt_name, bs_mul in zip(
            self.nets, self.opts, self.opt_names, self.bs_muls
        ):
            hist.append(
                {
                    "task_name": self.task_name,
                    "name": opt_name,
                    "bs_mul": bs_mul,
                    "lr_decay": self.lr_decay,
                    "train_cbl_loss": [],
                    "train_loss": [],
                    "train_x": [],
                    "val_cbl_loss": [],
                    "val_loss": [],
                    "val_x": [],
                    "train_acc_top_1": [],
                    "train_acc_top_5": [],
                    "val_acc_top_1": [],
                    "val_acc_top_5": [],
                    "test_acc_top_1": [],
                    "test_acc_top_5": [],
                    "norm_diffs": [],
                    "epochs_x": [],
                    "norm_diffs_x": [],
                    "total_steps": 0,
                    "prev_val_eval_step": 0,
                    "prev_grad_norms_eval_step": 0,
                    "batch_end": True,
                }
            )
        return hist

    def train(self, batch_mul_step_count=400, calc_norm_diffs=False):
        loading_path = self.task_name
        os.makedirs(loading_path, exist_ok=True)
        for epoch in range(self.num_epochs):
            for net, optimizer, net_hist in zip(self.nets, self.opts, self.hist):
                net.to(device)
                # optimizer_to(optimizer, device)
                optimizer_cbl, optimizer_head = optimizer
                optimizer_to(optimizer_cbl, device)
                optimizer_to(optimizer_head, device)

                total_steps = net_hist["total_steps"]
                bs_mul = net_hist["bs_mul"]
                lr_decay = net_hist["lr_decay"]

                if net_hist["bs_mul"] == "linear":
                    if not ("bs_mul_value" in net_hist):
                        net_hist["bs_mul_value"] = 1
                    bs_mul = net_hist["bs_mul_value"]

                net_hist["epochs_x"].append(total_steps)

                for i, data in enumerate(self.train_loader, 0):
                    prev_grad_norms_eval_step = net_hist["prev_grad_norms_eval_step"]

                    if (
                        calc_norm_diffs
                        and (
                            (total_steps - prev_grad_norms_eval_step)
                            > norm_diffs_step_count
                            or len(net_hist["norm_diffs"]) == 0
                        )
                        and net_hist["batch_end"]
                    ):
                        net_hist["prev_grad_norms_eval_step"] = total_steps
                        net.eval()
                        norm_diffs = compute_norm_diffs(
                            net,
                            optimizer_head,
                            criterion_head,
                            self.train_loader,
                            self.train_loader,
                            full_grad=None,
                            repeats=5,
                        )

                        net_hist["norm_diffs"].append(norm_diffs)
                        net_hist["norm_diffs_x"].append(total_steps)
                        net.train()

                    # net.train()
                    net_hist["batch_end"] = False

                    if lr_decay < 1:
                        for g in optimizer.param_groups:
                            g["lr"] = g["lr"] * lr_decay

                    # optimizer_cbl.zero_grad()
                    # optimizer_head.zero_grad()
                    inputs, labels = data
                    inputs, targets = inputs.to(device), torch.LongTensor(labels).to(
                        device
                    )
                    cbl_logits, logits = net(**inputs)

                    cbl_loss = self.criterion_cbl(cbl_logits)
                    cbl_loss.backward(retain_graph=True)

                    ce_loss = self.criterion_head(logits, targets)
                    ce_loss.backward()

                    # optimizer_cbl.step()
                    # optimizer_head.step()

                    if total_steps % bs_mul == bs_mul - 1:
                        optimizer_cbl.step()
                        optimizer_cbl.zero_grad()
                        optimizer_head.step()
                        optimizer_head.zero_grad()
                        net_hist["batch_end"] = True

                    net_hist["train_loss"].append(
                        ce_loss.detach().cpu().item() * bs_mul
                    )
                    net_hist["train_cbl_loss"].append(
                        cbl_loss.detach().cpu().item() * bs_mul
                    )
                    net_hist["train_x"].append(total_steps)

                    if total_steps % bs_mul == bs_mul - 1:
                        if net_hist["bs_mul"] == "linear":
                            net_hist["bs_mul_value"] = (
                                int(int(total_steps) / batch_mul_step_count) + 1
                            )
                            bs_mul = net_hist["bs_mul_value"]

                    top_1, top_5 = accuracy(logits, targets, topk=(1, 5))
                    net_hist["train_acc_top_1"].append(top_1.detach().cpu().item())
                    net_hist["train_acc_top_5"].append(top_5.detach().cpu().item())

                    prev_val_eval_step = net_hist["prev_val_eval_step"]
                    if (total_steps - prev_val_eval_step) > val_step_count and net_hist[
                        "batch_end"
                    ]:
                        net_hist["prev_val_eval_step"] = total_steps

                        net.eval()

                        val_cbl_losses = []
                        val_ce_losses = []
                        val_top_1_accs = []
                        val_top_5_accs = []

                        with torch.no_grad():
                            for step, val_data in enumerate(self.val_loader):
                                inputs, labels = val_data
                                inputs, targets = inputs.to(device), torch.LongTensor(
                                    labels
                                ).to(device)
                                cbl_logits, logits = net(**inputs)

                                cbl_loss = self.criterion_cbl(cbl_logits)
                                val_cbl_losses.append(cbl_loss.detach().cpu().item())

                                ce_loss = self.criterion_head(logits, targets)
                                val_ce_losses.append(ce_loss.detach().cpu().item())

                                top_1, top_5 = accuracy(logits, targets, topk=(1, 5))
                                val_top_1_accs.append(top_1.detach().cpu().item())
                                val_top_5_accs.append(top_5.detach().cpu().item())

                        net_hist["val_loss"].append(np.mean(val_ce_losses))
                        net_hist["val_x"].append(total_steps)
                        net_hist["val_cbl_loss"].append(np.mean(val_cbl_losses))

                        net_hist["val_acc_top_1"].append(np.mean(val_top_1_accs))
                        net_hist["val_acc_top_5"].append(np.mean(val_top_1_accs))

                        net.train()

                        if total_steps % 10 == 0:
                            display.clear_output(wait=True)

                            grouped_hist = group_uniques_full(
                                net_hist,
                                [
                                    "train_loss",
                                    "train_cbl_loss",
                                    "val_loss",
                                    "val_cbl_loss",
                                    "train_acc_top_1",
                                    "train_acc_top_5" "val_acc_top_1",
                                    "val_acc_top_5",
                                ],
                            )

                            fig = plt.figure(
                                figsize=(15, 8 + 2 * ((len(grouped_hist) + 2) // 3)),
                                dpi=300,
                            )
                            gs = GridSpec(
                                4 + 2 * ((len(grouped_hist) + 2) // 3), 3, figure=fig
                            )

                            ax1 = fig.add_subplot(gs[0:4, :2])
                            ax2 = fig.add_subplot(gs[0:2, 2])
                            ax3 = fig.add_subplot(gs[2:4, 2])

                            ax1 = make_loss_plot(
                                ax1,
                                grouped_hist,
                                loss_name="CE loss",
                                eps=0.01,
                                make_val=False,
                                alpha=0.9,
                            )
                            ax2 = make_accuracy_plot(
                                ax2,
                                grouped_hist,
                                eps=0.01,
                                make_train=True,
                                make_val=False,
                                top_k=1,
                                alpha=0.9,
                            )
                            ax3 = make_accuracy_plot(
                                ax3,
                                grouped_hist,
                                eps=0.01,
                                make_train=False,
                                make_val=True,
                                top_k=1,
                                alpha=0.9,
                            )

                            if calc_norm_diffs:
                                draw_norm_hists_for_different_models(
                                    fig,
                                    gs[4:, :],
                                    grouped_hist,
                                    bins_n=100,
                                    draw_normal=True,
                                )

                            gs.tight_layout(fig)
                            plt.draw()
                            plt.show()

                        total_steps += 1
                        net_hist["total_steps"] = total_steps

                # net_hist["total_steps"] = total_steps

            torch.save(
                {
                    "epoch": epoch,
                    f'{net_hist["val_acc_top_1"]}_model_state_dict': net.state_dict(),
                    "optimizer_cbl_state_dict": optimizer_cbl.state_dict(),
                    "optimizer_head_state_dict": optimizer_head.state_dict(),
                    "loss_train": net_hist["train_loss"],
                    "cbl_loss_train": net_hist["train_cbl_loss"],
                    "loss_val": net_hist["val_loss"],
                    "cbl_loss_val": net_hist["val_cbl_loss"],
                },
                os.path.join(
                    loading_path,
                    f'{net_hist["val_acc_top_1"]}_checkpoint_{epoch}_epoch.pth',
                ),
            )

        print("Finished Training")

    def test(self):
        for net, optimizer, net_hist in zip(self.nets, self.opts, self.hist):
            net.eval()
            test_top_1_accs, test_top_5_accs = [], []
            with torch.no_grad():
                for step, batch in enumerate(self.test_loader, 0):
                    inputs, labels = batch
                    inputs, targets = inputs.to(device), torch.LongTensor(labels).to(
                        device
                    )
                    cbl_logits, logits = net(**inputs)
                    top_1, top_5 = accuracy(logits, targets, topk=(1, 5))
                    test_top_1_accs.append(top_1.detach().cpu().item())
                    test_top_5_accs.append(top_5.detach().cpu().item())

            net_hist["test_acc_top_1"].append(np.mean(test_top_1_accs))
            net_hist["test_acc_top_5"].append(np.mean(test_top_5_accs))

        return {
            "Top 1 accuracy": net_hist["test_acc_top_1"],
            "Top 5 accuracy": net_hist["test_acc_top_5"],
        }
