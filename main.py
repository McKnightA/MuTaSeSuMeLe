import torch
import Datasets
import Tasks
import BackBoneModel as bbm
import MetaOptimizer as mo
import Tests


# todo:
#  (now) it breaks with data of a different image size.
#        Is this an issue with the model architecture? Yes.
#        Solution: make image size invariant model or resize datasets to match cifar10
#  (next) adjust all tests to accommodate general datasets
#  (with purchase of compute) test multi task training over long times on varying datasets
#  (next) MAML breaks for rotation and contrastive, so fix it
#  (after all tests are operational) integrate wandb to track test results
#  (eventually) there's code repetition for running each type of test with a taguchi array


dataset = Datasets.Food101()


embedding_size = 1024  # hyper parameters
batch_size = 4
pre_epochs = 3
inner_loops = 5
fine_epochs = 2
device = 'cpu'

# p=4, l=2  [0, 0, 0, 0],  # control = frozen random weights and pretraining classification
#
taguchi_array = [[0, 0, 0, 1],
                 [0, 1, 1, 0],
                 [0, 1, 1, 1],
                 [1, 0, 1, 0],
                 [1, 0, 1, 1],
                 [1, 1, 0, 0],
                 [1, 1, 0, 1]]


def spawn_tasklist():
    return [Tasks.Rotation(embedding_size, bbm.Cifar10Classifier, device),
            Tasks.Colorization(embedding_size, bbm.Cifar10Decoder, device),
            Tasks.Contrastive(embedding_size, bbm.Cifar10Classifier, device),
            Tasks.MaskedAutoEncoding(embedding_size, bbm.Cifar10Decoder, device),
            Tasks.Classification(embedding_size, bbm.Cifar10Classifier, 18, "food101", device)]


def spawn_backbone():
    return bbm.Cifar10Encoder(embedding_size, device)


backbone = spawn_backbone()
tasks = spawn_tasklist()

eval_task = Tasks.Classification(embedding_size, bbm.Cifar10Classifier, 18, "food101", device)
# perhaps dataset should hold the number of classes to avoid magic number 18

meta_task = Tasks.Cifar10Classification(embedding_size, bbm.Cifar10Classifier, device)

meta_optim = mo.MAML([tasks[0]], meta_task, backbone, 0.001, torch.optim.Adam)

# -----------------------------------------------------------------------------------

# Tests.test_task(dataset, testset, backbone, tasks[2], pre_epochs, batch_size)

"""
for trial in taguchi_array:
    trial_tasks = []

    del tasks
    tasks = spawn_tasklist()
    del backbone
    backbone = spawn_backbone()

    for variable in zip(trial, tasks):
        if variable[0]:
            trial_tasks.append(variable[1])

    if len(trial_tasks) == 0:
        baseline = True
        trial_tasks.append(tasks[-1])
    else:
        baseline = False

    multi_task = Tasks.AveragedLossMultiTask(trial_tasks, device)
    Tests.test_task(dataset, testset, backbone, multi_task, pre_epochs, batch_size, baseline)
"""

Tests.test_finetune(dataset, dataset, dataset, backbone, tasks[-1], eval_task, pre_epochs, fine_epochs, batch_size)

"""
for trial in taguchi_array:
    trial_tasks = []

    del tasks
    tasks = spawn_tasklist()
    del backbone
    backbone = spawn_backbone()

    for variable in zip(trial, tasks):
        if variable[0]:
            trial_tasks.append(variable[1])

    if len(trial_tasks) == 0:
        baseline = True
    else:
        baseline = False

    multi_task = Tasks.AveragedLossMultiTask(trial_tasks, device)
    Tests.test_finetune(dataset, dataset, dataset, backbone, multi_task, tasks[-1],
                        pre_epochs, fine_epochs, batch_size, baseline)
"""

# Tests.test_maml(dataset, dataset, testset,
#                 backbone, meta_optim, eval_task,
#                 pre_epochs, inner_loops, fine_epochs, batch_size)

"""
for trial in taguchi_array:
    print(trial)
    trial_tasks = []

    del tasks
    tasks = spawn_tasklist()
    del backbone
    backbone = spawn_backbone()

    for variable in zip(trial, tasks):
        if variable[0]:
            trial_tasks.append(variable[1])
    if len(trial_tasks) == 0:
        baseline = True

    else:
        baseline = False

    meta_optim = mo.MAML(trial_tasks, meta_task, backbone, 0.001, torch.optim.Adam)
    results.append(Tests.test_maml(dataset, dataset, testset,
                                   backbone, meta_optim, eval_task,
                                   pre_epochs, inner_loops, fine_epochs,
                                   batch_size))

meta_optim = mo.MAML([tasks[4]], meta_task, backbone, 0.001, torch.optim.Adam)
Tests.test_maml(dataset, dataset, testset, backbone, meta_optim, eval_task,
                pre_epochs, inner_loops, fine_epochs, batch_size)
"""
