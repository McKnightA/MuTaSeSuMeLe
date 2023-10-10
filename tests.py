import torch
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import Tasks
import BackBoneModel as bbm
import MetaOptimizer as mo


def test_maml(training_set, finetune_set, eval_set, common_backbone, meta_optimizer, finetune_task,
              pretrain_loops, inner_meta_loops, finetune_loops, batch_size):
    """

    :param training_set:
    :param finetune_set:
    :param eval_set:
    :param common_backbone:
    :param meta_optimizer:
    :param finetune_task:
    :param pretrain_loops:
    :param inner_meta_loops:
    :param finetune_loops:
    :param batch_size:
    :return:
    """

    pretrain_losses = []
    pretrain_eval_losses = []

    for j in range(pretrain_loops):  # pretraining loop
        print("epoch: ", j + 1)

        for batch in training_set.iter(batch_size):
            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0)
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            # forward pass the data
            loss = meta_optimizer.outer_loop((data_batch.transpose((0, 3, 1, 2)), labels), inner_meta_loops)

            pretrain_losses.append(loss.item())

        # this may not work if the meta_optim.test_task isn't the same as finetune_task
        for batch in eval_set.iter(batch_size):
            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated = meta_optim.test_task.pretreat(data_batch)  # forward pass the data
            embedded = common_backbone.forward(treated)
            try:
                loss = meta_optim.test_task.generate_loss(embedded)
            except TypeError:
                loss = meta_optim.test_task.generate_loss(embedded, labels)

            pretrain_eval_losses.append(loss.item())

    # fine-tune --------------------------------------------------------------------------------------------------------
    fine_optim = torch.optim.Adam(list(finetune_task.task_head.parameters()))

    fine_losses = []
    fine_eval_losses = []

    for i in range(finetune_loops):  # fine-tuning loop
        print("fine tuning epoch: ", i + 1)

        for batch in finetune_set.iter(batch_size):

            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated = finetune_task.pretreat(data_batch)
            embedded = common_backbone.forward(treated)  # forward pass the data
            try:
                loss = finetune_task.generate_loss(embedded)
            except TypeError:
                loss = finetune_task.generate_loss(embedded, labels)
            fine_losses.append(loss.item())

            fine_optim.zero_grad()  # backward pass the loss
            loss.backward()
            fine_optim.step()

        for batch in eval_set.iter(batch_size):
            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated = finetune_task.pretreat(data_batch)
            embedded = common_backbone.forward(treated)  # forward pass the data
            try:
                loss = finetune_task.generate_loss(embedded)
            except TypeError:
                loss = finetune_task.generate_loss(embedded, labels)
            fine_eval_losses.append(loss.item())

    # visualization ----------------------------------------------------------------------------------------------------
    plt.plot(pretrain_losses)
    plt.plot(pretrain_eval_losses)
    plt.savefig("plots/{} training : {} meta.png".format(tasks[0].name, meta_task.name))
    plt.plot(fine_losses)
    plt.plot(fine_eval_losses)
    plt.title("{} training : {} meta | pretrain :: {} | finetune".format(tasks[0].name, meta_task.name, eval_task.name))
    plt.savefig("plots/meta_multi_ssl_pretraining.png")


def test_finetune(training_set, finetune_set, eval_set, common_backbone, pretrain_task, finetune_task,
                  pretrain_loops, finetune_loops, batch_size):
    """

    :param training_set:
    :param finetune_set:
    :param eval_set:
    :param common_backbone:
    :param pretrain_task:
    :param finetune_task:
    :param pretrain_loops:
    :param finetune_loops:
    :param batch_size:
    :return:
    """
    try:  # some tasks require data modification that requires "harmonization" with the original data format
        params = list(pretrain_task.harmonization.parameters()) + list(common_backbone.parameters()) \
                 + list(pretrain_task.task_head.parameters())
    except AttributeError:
        params = list(common_backbone.parameters()) + list(pretrain_task.task_head.parameters())
    optim = torch.optim.Adam(params)

    # pretraining ------------------------------------------------------------------------------------------------------
    pretrain_losses = []
    for i in range(pretrain_loops):
        print("epoch: ", i + 1)

        for batch in training_set.iter(batch_size):

            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated_batch = pretrain_task.pretreat(data_batch)
            latent = common_backbone.forward(treated_batch)
            try:
                loss = pretrain_task.generate_loss(latent)
            except TypeError:
                loss = pretrain_task.generate_loss(latent, labels)
            pretrain_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()

    # fine-tuning ------------------------------------------------------------------------------------------------------
    fine_optim = torch.optim.Adam(list(finetune_task.task_head.parameters()))

    fine_losses = []
    fine_eval_losses = []

    for i in range(finetune_loops):
        print("fine-tuning epoch: ", i + 1)

        for batch in finetune_set.iter(batch_size):
            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated_batch = finetune_task.pretreat(data_batch)
            embedded = common_backbone.forward(treated_batch)
            try:
                loss = finetune_task.generate_loss(embedded)
            except TypeError:
                loss = finetune_task.generate_loss(embedded, labels)
            fine_losses.append(loss.item())

            fine_optim.zero_grad()
            loss.backward()
            fine_optim.step()

        for batch in eval_set.iter(batch_size):
            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated_batch = finetune_task.pretreat(data_batch)
            embedded = common_backbone.forward(treated_batch)
            try:
                loss = finetune_task.generate_loss(embedded)
            except TypeError:
                loss = finetune_task.generate_loss(embedded, labels)
            fine_eval_losses.append(loss.item())

    # plt.plot(pretrain_losses)
    # plt.plot(pretrain_eval_losses)
    # plt.show()
    plt.plot(fine_losses)
    plt.plot(fine_eval_losses)
    plt.title("fine tuning on {} with {} pretraining".format(finetune_task.name, pretrain_task.name))
    plt.savefig("plots/{}_pretraining.png".format(pretrain_task.name))


def test_task(training_set, eval_set, common_backbone, task, training_loops, batch_size):
    """

    :param training_set:
    :param eval_set:
    :param common_backbone:
    :param task:
    :param training_loops:
    :param batch_size:
    :return:
    """
    try:
        params = list(task.harmonization.parameters()) + list(common_backbone.parameters()) \
                 + list(task.task_head.parameters())
    except AttributeError:
        params = list(common_backbone.parameters()) + list(task.task_head.parameters())
    optim = torch.optim.Adam(params)

    losses = []
    eval_losses = []

    for i in range(training_loops):
        print("epoch: ", i + 1)
        for batch in training_set.iter(batch_size):

            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated = task.pretreat(data_batch)  # forward pass the data
            embedded = common_backbone.forward(treated)
            try:
                loss = task.generate_loss(embedded)
            except TypeError:
                loss = task.generate_loss(embedded, labels)
            losses.append(loss.item())

            optim.zero_grad()  # backward pass the loss
            loss.backward()
            optim.step()

        for batch in eval_set.iter(batch_size):

            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            treated = task.pretreat(data_batch)  # forward pass the data
            embedded = common_backbone.forward(treated)
            try:
                loss = task.generate_loss(embedded)
            except TypeError:
                loss = task.generate_loss(embedded, labels)

            eval_losses.append(loss.item())

    plt.semilogy(losses)
    plt.semilogy(eval_losses)
    plt.show()


if __name__ == "__main__":

    dataset = load_dataset("cifar10", split="train", streaming=True)
    testset = load_dataset("cifar10", split='test', streaming=True)

    embedding_size = 128  # hyper parameters
    batch_size = 32
    pre_epochs = 5
    inner_loops = 5
    fine_epochs = 2
    device = "cpu"

    backbone = bbm.SimpleConvEncode(embedding_size, device)

    tasks = [Tasks.Rotation(embedding_size, bbm.SimpleTaskHead, device),
             Tasks.Colorization(embedding_size, bbm.SimpleConvDecode, device),
             Tasks.Contrastive(embedding_size, bbm.SimpleTaskHead, device),
             Tasks.MaskedAutoEncoding(embedding_size, bbm.SimpleConvDecode, device),
             Tasks.Cifar10Classification(embedding_size, bbm.SimpleTaskHead, device)]

    eval_task = Tasks.Cifar10Classification(embedding_size, bbm.SimpleTaskHead, device)

    meta_task = Tasks.Cifar10Classification(embedding_size, bbm.SimpleTaskHead, device)

    meta_optim = mo.MAML([tasks[0]], meta_task, backbone, 0.001, torch.optim.Adam)

    test_task(dataset, testset, backbone, tasks[0], pre_epochs, batch_size)

    # test_finetune(dataset, dataset, testset, backbone, tasks[0], eval_task, pre_epochs, fine_epochs, batch_size)

    # test_maml(dataset, dataset, testset,
    #           backbone, meta_optim, eval_task,
    #           pre_epochs, inner_loops, fine_epochs, batch_size)
