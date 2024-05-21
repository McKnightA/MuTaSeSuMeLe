import wandb
import torch
import Datasets
import Tasks
import BackBoneModel as bbm
import MetaOptimizer as mo
import Tests

"""
Hypothesis: if a model is trained with MAML and multi self supervised tasks are used as the inner loop tasks,
             and the desired task is used as the outerloop, then the model will achieve better results in fewer
             epochs than a model that is pretrained on imagenet classification and fine tuned or linearly probed on the desired task
"""

# todo: implements
#  (now) create hypothesis test
#  (next) adjust MAML in a manner conducive to testing the hypothesis
#  (then) explore other datasets -- https://huggingface.co/datasets/detection-datasets/coco
#                                   requires a new task for object detection   D task
#  (then) create a resnet projector
#                                -- also look at Pascal VOC, ADE20K for segmentation   S task
#  (then) integrate wandb and save loss values not just plots
#  (then) explore other SSL methods -- information maxing looks really interesting https://arxiv.org/abs/2105.04906
#                                   -- maybe a clustering method https://arxiv.org/pdf/2006.09882
#  (then) make next architecture
#  (with purchase of compute) test over long times -- ~400 SSL epochs is common though some only use 100
#  (eventually) there's code repetition for running each different type of test with a taguchi array

# todo: systems tests
#  (then) debug improved multi task results
#  (then) debug ResNet50 architecture on colab
#  (then) debug imagenet data and label processing on colab

# todo: research tests
#  (now) collect early meta learning results

dataset = Datasets.Cifar10()
testset = Datasets.Cifar10()

# hyper parameters
embedding_size = 1024
batch_size = 64
pre_epochs = 10
inner_loops = 1
fine_epochs = 10
device = 'cpu'

config = {"embedding size": embedding_size,
          "batch size": batch_size,
          "pretrain epochs": pre_epochs,
          "finetune epochs": fine_epochs,
          "device": device,
          "pretrain dataset": dataset.name,
          "finetune dataset": testset.name}

# p=4, l=2  [0, 0, 0, 0],  # control = frozen random weights
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
            Tasks.Classification(embedding_size, bbm.Cifar10Classifier, testset.class_num, testset.name, device)]


def spawn_backbone():
    return bbm.Cifar10Encoder(embedding_size, device)


backbone = spawn_backbone()
tasks = spawn_tasklist()

train_task = Tasks.Classification(embedding_size, bbm.Cifar10Classifier, dataset.class_num, dataset.name, device)

eval_task = Tasks.Classification(embedding_size, bbm.Cifar10Classifier, testset.class_num, testset.name, device)

meta_task = Tasks.Classification(embedding_size, bbm.Cifar10Classifier, testset.class_num, testset.name, device)

meta_optim = mo.MAML([tasks[0], tasks[2]], meta_task, backbone, 0.001, torch.optim.Adam)

# -----------------------------------------------------------------------------------

# Tests.test_task(dataset, dataset, backbone, tasks[4], pre_epochs, batch_size, baseline=False)

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

# Tests.test_finetune(dataset, testset, testset, backbone, train_task, eval_task, pre_epochs, fine_epochs, batch_size)

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
    Tests.test_finetune(dataset, testset, testset, backbone, multi_task, tasks[-1],
                        pre_epochs, fine_epochs, batch_size)
"""

Tests.test_maml_pretraining(dataset, testset, backbone,
                            meta_optim, eval_task, pre_epochs,
                            inner_loops, fine_epochs, batch_size)

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
