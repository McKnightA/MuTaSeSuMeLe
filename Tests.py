import torch
import numpy as np
import matplotlib.pyplot as plt


# TODO: finish hypothesis test

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
    # performances = []
    eval_losses = []
    # eval_performances = []

    for i in range(training_loops):
        print("epoch: ", i + 1)

        average_loss = np.average(run_data_loop(training_set, task, common_backbone, optim, batch_size, True))
        losses.append(average_loss)

        average_loss = np.average(run_data_loop(eval_set, task, common_backbone, optim, batch_size, False))
        eval_losses.append(average_loss)

    fig = plt.figure()
    plt.plot(losses, label="training")
    plt.plot(eval_losses, label="eval")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()

    if baseline:
        title = "{} baseline loss".format(task.name)
    else:
        title = "{} loss".format(task.name)

    plt.title(title)
    plt.savefig("plots/" + title)

    """try:
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
        plt.savefig("plots/{} performance".format(task.name))"""


def test_finetune(training_set, finetune_set, common_backbone, pretrain_task, finetune_task,
                  pretrain_loops, finetune_loops, batch_size):
    """
    finetuning according to https://snorkel.ai/boost-foundation-model-results-with-linear-probing-fine-tuning/
    :param training_set:
    :param finetune_set:
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
    pretrain_eval_losses = []
    for i in range(pretrain_loops):
        print("epoch: ", i + 1)

        average_loss = np.average(run_data_loop(training_set, pretrain_task, common_backbone, optim, batch_size, True))
        pretrain_losses.append(average_loss)

        average_loss = np.average(run_data_loop(training_set, pretrain_task, common_backbone, optim, batch_size, False))
        pretrain_eval_losses.append(average_loss)

    results.append(pretrain_losses)
    results.append(pretrain_eval_losses)

    # fine-tuning ------------------------------------------------------------------------------------------------------
    fine_optim = torch.optim.Adam(list(finetune_task.parameters()) + list(common_backbone.parameters()))

    fine_losses = []
    fine_eval_losses = []

    for i in range(finetune_loops):
        print("fine-tuning epoch: ", i + 1)

        averaged_loss = np.mean(run_data_loop(finetune_set, finetune_task, common_backbone,
                                              fine_optim, batch_size, True))
        fine_losses.append(averaged_loss)

        averaged_loss = np.mean(run_data_loop(finetune_set, finetune_task, common_backbone,
                                              fine_optim, batch_size, False))
        fine_eval_losses.append(averaged_loss)

    results.append(fine_losses)
    results.append(fine_eval_losses)

    # saving results ---------------------------------------------------------------------------------------------------
    fig = plt.figure()
    plt.plot(pretrain_losses, label="training")
    plt.plot(pretrain_eval_losses, label="eval")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()
    plt.title("{} pretrain training loss".format(pretrain_task.name))
    plt.savefig("plots/{} pretrain training loss.png".format(pretrain_task.name))

    fig = plt.figure()
    plt.plot(fine_losses, label="training")
    plt.plot(fine_eval_losses, label="eval")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()
    plt.title("{} pretrain _ {} finetune".format(pretrain_task.name, finetune_task.name))
    plt.savefig("plots/{} pretraining with fine tuning on {}.png".format(pretrain_task.name, finetune_task.name))

    return results


def test_linear_probe(training_set, probing_sets, common_backbone, pretrain_task, probing_tasks,
                      pretrain_loops, finetune_loops, batch_size):
    """

    :param training_set:
    :param probing_sets: a list of datasets for probing the feature extractor must match with the probe tasks
    :param common_backbone:
    :param pretrain_task:
    :param probing_tasks: a list of tasks to probe the feature extractor (common backbone) with
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
    pretrain_eval_losses = []
    for i in range(pretrain_loops):
        print("epoch: ", i + 1)

        average_loss = np.average(run_data_loop(training_set, pretrain_task, common_backbone, optim, batch_size, True))
        pretrain_losses.append(average_loss)

        average_loss = np.average(run_data_loop(training_set, pretrain_task, common_backbone, optim, batch_size, False))
        pretrain_eval_losses.append(average_loss)

    results.append(pretrain_losses)

    # saving results ---------------------------------------------------------------------------------------------------
    fig = plt.figure()
    plt.plot(pretrain_losses, label="training")
    plt.plot(pretrain_eval_losses, label="eval")
    plt.xlabel("epochs")
    plt.ylabel("averaged loss")
    plt.legend()
    plt.title("{} pretrain training loss".format(pretrain_task.name))
    plt.savefig("plots/{} pretrain training loss.png".format(pretrain_task.name))

    # probing ----------------------------------------------------------------------------------------------------------
    for probe_set, probe_task in zip(probing_sets, probing_tasks):
        fine_optim = torch.optim.Adam(list(probe_task.parameters()))

        probe_losses = []
        probe_eval_losses = []

        for i in range(finetune_loops):
            print("fine-tuning epoch: ", i + 1)

            averaged_loss = np.mean(run_data_loop(probe_set, probe_task, common_backbone,
                                                  fine_optim, batch_size, True))
            probe_losses.append(averaged_loss)

            averaged_loss = np.mean(run_data_loop(probe_set, probe_task, common_backbone,
                                                  fine_optim, batch_size, False))
            probe_eval_losses.append(averaged_loss)

        results.append(probe_losses)
        results.append(probe_eval_losses)

        # saving results -----------------------------------------------------------------------------------------------
        fig = plt.figure()
        plt.plot(probe_losses, label="training")
        plt.plot(probe_eval_losses, label="eval")
        plt.xlabel("epochs")
        plt.ylabel("averaged loss")
        plt.legend()
        plt.title("{} pretraining - probing {}".format(pretrain_task.name, probe_task.name))
        plt.savefig("plots/{} pretraining - probing {}.png".format(pretrain_task.name, probe_task.name))

    return results


def test_maml_pretraining(training_set, finetune_set, common_backbone, meta_optimizer, finetune_task,
                          pretrain_loops, inner_meta_loops, finetune_loops, batch_size):
    """

    :param training_set:
    :param finetune_set:
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

        batch_losses = []
        batch_eval_losses = []
        for batch in training_set.trainset.iter(batch_size):
            data, labels = training_set.get_data_n_labels(batch)

            # forward pass the data
            loss = meta_optimizer.outer_loop((data, labels), inner_meta_loops)

            batch_losses.append(loss.item())

        for batch in training_set.testset.iter(batch_size):
            data, labels = training_set.get_data_n_labels(batch)

            try:
                loss = meta_optimizer.test_task.forward(data, common_backbone)
            except TypeError:
                loss = meta_optimizer.test_task.forward(data, labels, common_backbone)

            batch_eval_losses.append(loss.item())

        pretrain_losses.append(np.mean(batch_losses))
        pretrain_eval_losses.append(np.mean(batch_eval_losses))

    # fine-tune --------------------------------------------------------------------------------------------------------
    fine_optim = torch.optim.Adam(list(finetune_task.parameters()))

    fine_losses = []
    fine_eval_losses = []

    for i in range(finetune_loops):  # fine-tuning loop
        print("fine tuning epoch: ", i + 1)

        averaged_loss = np.mean(run_data_loop(finetune_set, finetune_task, common_backbone,
                                              fine_optim, batch_size, True))
        fine_losses.append(averaged_loss)

        averaged_loss = np.mean(run_data_loop(finetune_set, finetune_task, common_backbone,
                                              fine_optim, batch_size, False))
        fine_eval_losses.append(averaged_loss)

    # visualization ----------------------------------------------------------------------------------------------------
    fig1 = plt.figure()
    plt.plot(pretrain_losses, label="meta training")
    plt.plot(pretrain_eval_losses, label="meta eval")
    plt.legend()
    task_name = ""
    for task in meta_optimizer.tasks:
        task_name += task.name + " "
    plt.title("{} training | {} meta".format(task_name, meta_optimizer.test_task.name))
    plt.savefig("plots/{}_training-{}_meta.png".format(task_name, meta_optimizer.test_task.name))

    fig2 = plt.figure()
    plt.plot(fine_losses, label="training")
    plt.plot(fine_eval_losses, label="eval")
    plt.legend()
    plt.title("{} training | {} meta pretrain || {} finetune".format(task_name,
                                                                     meta_optimizer.test_task.name,
                                                                     finetune_task.name))
    plt.savefig("plots/{}_training-{}_meta---{}_finetune.png".format(task_name,
                                                                     meta_optimizer.test_task.name,
                                                                     finetune_task.name))

    results = [pretrain_losses, pretrain_eval_losses, fine_losses, fine_eval_losses]

    return results


def test_hypothesis(training_set, eval_set, common_backbone, meta_optimizer, eval_task,
                    eval_loops, inner_meta_loops, batch_size):
    """

    :param training_set:
    :param eval_set:
    :param common_backbone:
    :param meta_optimizer:
    :param eval_task:
    :param eval_loops:
    :param inner_meta_loops:
    :param batch_size:
    :return:
    """

    pretrain_losses = []
    pretrain_eval_losses = []

    for j in range(eval_loops):  # pretraining loop
        print("epoch: ", j + 1)

        for batch in training_set.trainset.iter(batch_size):
            data, labels = training_set.get_data_n_labels(batch)

            # forward pass the data
            loss = meta_optimizer.outer_loop((data, labels), inner_meta_loops)

            pretrain_losses.append(loss.item())

        for batch in training_set.testset.iter(batch_size):
            data, labels = training_set.get_data_n_labels(batch)

            try:
                loss = meta_optimizer.test_task.forward(data, common_backbone)
            except TypeError:
                loss = meta_optimizer.test_task.forward(data, labels, common_backbone)

            pretrain_eval_losses.append(loss.item())

    # visualization ----------------------------------------------------------------------------------------------------
    fig1 = plt.figure()
    plt.plot(pretrain_losses)
    plt.plot(pretrain_eval_losses)
    task_name = ""
    for task in meta_optimizer.tasks:
        task_name += task.name + " "
    plt.title("{} training | {} meta".format(task_name, meta_optimizer.test_task.name))
    plt.savefig("plots/{}_training-{}_meta.png".format(task_name, meta_optimizer.test_task.name))

    results = {}
    return results
