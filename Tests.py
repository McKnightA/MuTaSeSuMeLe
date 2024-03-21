import torch
import numpy as np
import matplotlib.pyplot as plt


# TODO:
#  (next) update test_task and test_maml
#  (then) fix multi_task_test_suite

def run_data_loop(dataset, task, backbone, optimizer, batch_size, training=False):
    losses = []

    if training:
        partition = dataset.trainset
    else:
        partition = dataset.testset

    for batch in partition.iter(batch_size):

        data_batch, labels = dataset.get_data_n_labels(batch)

        try:
            loss = task.forward(data_batch, backbone)
        except TypeError:
            loss = task.forward(data_batch, labels, backbone)

        losses.append(loss.item())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return losses


def test_task(training_set, eval_set, common_backbone, task, training_loops, batch_size, baseline=False):
    """

    :param training_set:
    :param eval_set:
    :param common_backbone:
    :param task:
    :param training_loops:
    :param batch_size:
    :param baseline:
    :return:
    """

    if baseline:  # training to fit a frozen backbone
        params = list(task.parameters())
    else:
        params = list(task.parameters()) + list(common_backbone.parameters())

    optim = torch.optim.Adam(params)
    # schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim, training_loops)

    losses = []
    performances = []
    eval_losses = []
    eval_performances = []

    for i in range(training_loops):
        print("epoch: ", i + 1)

        batch_losses = []
        batch_performance = []
        for batch in training_set.iter(batch_size):

            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            try:
                loss = task.forward(data_batch, common_backbone)
                batch_performance.append(task.check_performance(data_batch, common_backbone))
            except TypeError:
                loss = task.foward(data_batch, labels, common_backbone)
                batch_performance.append(task.check_performance(data_batch, labels, common_backbone))

            batch_losses.append(loss.item())

            optim.zero_grad()  # backward pass the loss
            loss.backward()
            optim.step()
        # schedule.step()

        losses.append(np.mean(batch_losses))
        performances.append(batch_performance[-2])

        batch_eval_losses = []
        batch_eval_performance = []
        for batch in eval_set.iter(batch_size):

            data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose(
                (0, 3, 1, 2))
            labels = torch.as_tensor(batch['label'], dtype=torch.long)

            try:
                loss = task.forward(data_batch, common_backbone)
                batch_eval_performance.append(task.check_performance(data_batch, common_backbone))
            except TypeError:
                loss = task.forward(data_batch, labels, common_backbone)
                batch_eval_performance.append(task.check_performance(data_batch, labels, common_backbone))

            batch_eval_losses.append(loss.item())

        eval_losses.append(np.mean(batch_eval_losses))
        eval_performances.append(batch_eval_performance[-2])

    fig = plt.figure()
    plt.plot(losses, label="training")
    plt.plot(eval_losses, label="eval")
    plt.title("{} loss".format(task.name))
    plt.legend()
    plt.savefig("plots/{} loss".format(task.name))

    try:
        fig = plt.figure()
        plt.plot(performances, label="training")
        plt.plot(eval_performances, label="eval")
        plt.title("{} performance".format(task.name))
        plt.legend()
        plt.savefig("plots/{} performance".format(task.name))

    except ValueError:
        fig, ax = plt.subplots(len(performances), 2)
        for i, img, eval_img in enumerate(zip(performances, eval_performances)):
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(eval_img)

        plt.title("{} performance".format(task.name))
        plt.legend()
        plt.savefig("plots/{} performance".format(task.name))


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
    results = []

    # pretraining ------------------------------------------------------------------------------------------------------
    params = list(pretrain_task.parameters()) + list(common_backbone.parameters())

    optim = torch.optim.Adam(params)

    pretrain_losses = []
    for i in range(pretrain_loops):
        print("epoch: ", i + 1)

        average_loss = np.average(run_data_loop(training_set, pretrain_task, common_backbone, optim, batch_size, True))
        pretrain_losses.append(average_loss)

    results.append(pretrain_losses)

    # fine-tuning ------------------------------------------------------------------------------------------------------
    fine_optim = torch.optim.Adam(list(finetune_task.parameters()))

    fine_losses = []
    fine_eval_losses = []

    for i in range(finetune_loops):
        print("fine-tuning epoch: ", i + 1)

        averaged_loss = np.mean(run_data_loop(finetune_set, finetune_task, common_backbone, fine_optim, batch_size, True))
        fine_losses.append(averaged_loss)

        averaged_loss = np.mean(run_data_loop(eval_set, finetune_task, common_backbone, fine_optim, batch_size, False))
        fine_eval_losses.append(averaged_loss)

    results.append(fine_losses)
    results.append(fine_eval_losses)

    # saving results ---------------------------------------------------------------------------------------------------
    fig = plt.figure()
    plt.plot(pretrain_losses)
    plt.title("{} pretrain training loss".format(pretrain_task.name))
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.savefig("plots/{} pretrain training loss.png".format(pretrain_task.name))

    fig = plt.figure()
    plt.plot(fine_losses)
    plt.plot(fine_eval_losses)
    plt.title("{} pretraining with fine tuning on {}".format(pretrain_task.name, finetune_task.name))
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.savefig("plots/{} pretraining with fine tuning on {}.png".format(pretrain_task.name, finetune_task.name))

    return results


def test_maml(training_set, finetune_set, eval_set, common_backbone, meta_optimizer, finetune_task,
              pretrain_loops, inner_meta_loops, finetune_loops, batch_size, visualize=False):
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
    :param visualize:
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

            treated = meta_optimizer.test_task.pretreat(data_batch)  # forward pass the data
            embedded = common_backbone.forward(treated)
            try:
                loss = meta_optimizer.test_task.generate_loss(embedded)
            except TypeError:
                loss = meta_optimizer.test_task.generate_loss(embedded, labels)

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
    if visualize:
        fig1 = plt.figure()
        plt.plot(pretrain_losses)
        plt.plot(pretrain_eval_losses)
        task_name = ""
        for task in meta_optimizer.tasks:
            task_name += task.name + " "
        plt.title("{} training : {} meta".format(task_name, meta_optimizer.test_task.name))
        plt.savefig("plots/{} training : {} meta.png".format(task_name, meta_optimizer.test_task.name))

        fig2 = plt.figure()
        plt.plot(fine_losses)
        plt.plot(fine_eval_losses)
        plt.title("{} training : {} meta | pretrain :: {} | finetune".format(task_name,
                                                                             meta_optimizer.test_task.name,
                                                                             finetune_task.name))
        plt.savefig("plots/{} training : {} meta | pretrain :: {} | finetune.png".format(task_name,
                                                                                         meta_optimizer.test_task.name,
                                                                                         finetune_task.name))
