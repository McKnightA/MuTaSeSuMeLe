import torch
import numpy as np
from Datasets import Cifar
from datasets import load_dataset
import matplotlib.pyplot as plt
import DataAugments as da
import Tasks
import BackBoneModel as bbm
import MetaOptimizer as mo
import torchmetrics
from skimage.color import rgb2lab, lab2rgb


# todo:
#  test huggingface dataset

file = "C:/Users/18147/Documents/Classes/2023 spring/DL/pa1/pa1/datasets/cifar-10-batches-py"
c = Cifar(file)
data = c.get_CIFAR10_data(subtract_mean=False)
print(data["X_train"].shape)

# testing meta-training with huggingface dataset

dataset = load_dataset("cifar10", split="train", streaming=True)
testset = load_dataset("cifar10", split='test', streaming=True)

latent_embedding_size = 128  # hyper parameters
batch_size = 32
pre_epochs = 5
inner_loops = 5
fine_epochs = 1

tasks = [Tasks.Colorization3(latent_embedding_size, bbm.SimpleConvDecode)]  # defining model objects
         # Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead),
         # Tasks.Contrastive4(latent_embedding_size, bbm.SimpleTaskHead),
         # Tasks.MaskedAutoEncoding(latent_embedding_size, bbm.SimpleConvDecode)]

meta_task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)

eval_task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)
backbone = bbm.SimpleConvEncode(latent_embedding_size)

meta_optim = mo.MAML(tasks, meta_task, backbone, 0.001, torch.optim.Adam)

# pretraining ----------------------------------------------------------------------------------------------------------
pretrain_losses = []
pretrain_eval_losses = []

for j in range(pre_epochs):  # pretraining loop
    print("epoch: ", j + 1)

    for batch in dataset.iter(batch_size):
        # print("loop: ", i)
        data_batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0)
        try:  # need labels if it's a supervised task
            labels = torch.as_tensor(batch['label'], dtype=torch.long)
        except KeyError:
            labels = None
            # carry on
        # forward pass the data
        loss = meta_optim.outer_loop((data_batch.transpose((0, 3, 1, 2)), labels), inner_loops)

        pretrain_losses.append(loss.item())

# fine-tune on classification ------------------------------------------------------------------------------------------
fine_optim = torch.optim.Adam(list(eval_task.task_head.parameters()))

fine_losses = []
fine_eval_losses = []

for j in range(fine_epochs):  # fine-tuning loop
    print("fine tuning epoch: ", j)

    for batch in dataset.iter(batch_size):
        # print("loop: ", i)
        batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose((0, 3, 1, 2))
        labels = torch.as_tensor(batch['labels'], dtype=torch.long)

        batch = eval_task.pretreat(batch)
        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_losses.append(loss.item())

        fine_optim.zero_grad()  # backward pass the loss
        loss.backward()
        fine_optim.step()

    for batch in testset.iter(batch_size):
        batch = np.concatenate([np.expand_dims(img, axis=0) for img in batch['img']], axis=0).transpose((0, 3, 1, 2))
        labels = torch.as_tensor(batch['labels'], dtype=torch.long)

        eval_task.pretreat(batch)
        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_eval_losses.append(loss.item())

# visualization --------------------------------------------------------------------------------------------------------
plt.plot(pretrain_losses)
plt.plot(pretrain_eval_losses)
plt.show()
plt.plot(fine_losses)
plt.plot(fine_eval_losses)
plt.title("col training : classification meta | pretrain :: classification | finetune")
plt.savefig("plots/meta_multi_ssl_pretraining.png")


# testing meta-training loop
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32
pre_epochs = 5
inner_loops = 5
fine_epochs = 1

tasks = [Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead),  # defining model objects
         Tasks.Colorization3(latent_embedding_size, bbm.SimpleConvDecode)]
         # Tasks.Contrastive4(latent_embedding_size, bbm.SimpleTaskHead),
         # Tasks.MaskedAutoEncoding(latent_embedding_size, bbm.SimpleConvDecode)]

meta_task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)

eval_task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)
backbone = bbm.SimpleConvEncode(latent_embedding_size)

meta_optim = mo.MAML(tasks, meta_task, backbone, 0.001, torch.optim.Adam)

# pretraining ----------------------------------------------------------------------------------------------------------
pretrain_losses = []
pretrain_eval_losses = []
indices = np.arange(data["X_train"].shape[0])
for j in range(pre_epochs):  # pretraining loop
    print("epoch: ", j + 1)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        data_batch = data["X_train"][indices[i - batch_size:i]]
        try:  # need labels if it's a supervised task
            labels = torch.as_tensor(data["y_train"][indices[i - batch_size:i]], dtype=torch.long)
        except KeyError:
            labels = None
            # carry on
        # forward pass the data
        loss = meta_optim.outer_loop((data_batch, labels), inner_loops)

        pretrain_losses.append(loss.item())

# fine-tune on classification ------------------------------------------------------------------------------------------
fine_optim = torch.optim.Adam(list(eval_task.task_head.parameters()))

fine_losses = []
fine_eval_losses = []

indices = np.arange(data["X_train"].shape[0])
for j in range(fine_epochs):  # fine-tuning loop
    print("fine tuning epoch: ", j)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        batch = eval_task.pretreat(data["X_train"][indices[i - batch_size:i]])
        labels = torch.as_tensor(data["y_train"][indices[i - batch_size:i]], dtype=torch.long)

        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_losses.append(loss.item())

        fine_optim.zero_grad()  # backward pass the loss
        loss.backward()
        fine_optim.step()

    for i in range(batch_size, data["X_val"].shape[0], batch_size):
        batch = eval_task.pretreat(data["X_val"][i - batch_size:i])
        labels = torch.as_tensor(data["y_val"][i - batch_size:i], dtype=torch.long)

        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_eval_losses.append(loss.item())

# visualization --------------------------------------------------------------------------------------------------------
plt.plot(pretrain_losses)
plt.plot(pretrain_eval_losses)
plt.show()
plt.plot(fine_losses)
plt.plot(fine_eval_losses)
plt.title("rot, col training : classification meta | pretrain :: classification | finetune")
plt.savefig("plots/meta_multi_ssl_pretraining.png")
"""

# test multitask self supervision by averaging
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32
pre_epochs = 5
fine_epochs = 1  # n-shot setting

tasks = [Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead),  # defining model objects
         Tasks.Colorization3(latent_embedding_size, bbm.SimpleConvDecode),
         Tasks.Contrastive4(latent_embedding_size, bbm.SimpleTaskHead),
         Tasks.MaskedAutoEncoding(latent_embedding_size, bbm.SimpleConvDecode)]

eval_task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)
backbone = bbm.SimpleConvEncode(latent_embedding_size)

params = list(backbone.parameters())  # list of Parameter objects
for task in tasks:
    try:  # some tasks require data modification that requires "harmonization" with the original data format
        params += list(task.harmonization.parameters()) + list(task.task_head.parameters())
    except AttributeError:
        params += list(task.task_head.parameters())
optim = torch.optim.Adam(params)

pretrain_losses = []
pretrain_eval_losses = []
indices = np.arange(data["X_train"].shape[0])
for j in range(pre_epochs):  # pretraining loop
    print("epoch: ", j + 1)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        try:  # need labels if it's a supervised task
            labels = torch.as_tensor(data["y_train"][indices[i - batch_size:i]], dtype=torch.long)
        except KeyError:
            continue
            # carry on
        # forward pass the data
        batch = [task.pretreat(data["X_train"][indices[i - batch_size:i]]) for task in tasks]

        latent = backbone.forward(torch.concatenate(batch, dim=0))

        total_loss = []
        first_index = 0  # tasks like contrastive change the batch size, so I count through each batch length
        for k in range(len(batch)):
            try:
                loss = tasks[k].generate_loss(latent[first_index:first_index + batch[k].shape[0]])
            except TypeError:  # ready for a supervised training task
                loss = tasks[k].generate_loss(latent[first_index:first_index + batch[k].shape[0]],
                                              labels)
            total_loss.append(loss)

            first_index += batch[k].shape[0]

        averaged_task_loss = torch.mean(torch.tensor(total_loss, requires_grad=True))
        pretrain_losses.append(averaged_task_loss.item())

        optim.zero_grad()  # backward pass the loss
        averaged_task_loss.backward()
        optim.step()

    for i in range(batch_size, data["X_val"].shape[0], batch_size):
        try:  # need labels if it's a supervised task
            labels = torch.as_tensor(data["y_val"][i - batch_size:i], dtype=torch.long)
        except KeyError:
            continue
            # carry on
        # forward pass the data
        batch = [task.pretreat(data["X_val"][i - batch_size:i]) for task in tasks]

        latent = backbone.forward(torch.concatenate(batch, dim=0))

        total_loss = []
        first_index = 0  # tasks like contrastive change the batch size, so I count through each batch length
        for k in range(len(batch)):
            try:
                loss = tasks[k].generate_loss(latent[first_index:first_index + batch[k].shape[0]])
            except TypeError:  # ready for a supervised training task
                loss = tasks[k].generate_loss(latent[first_index:first_index + batch[k].shape[0]],
                                              labels)
            total_loss.append(loss)

            first_index += batch[k].shape[0]

        averaged_task_loss = torch.mean(torch.tensor(total_loss, requires_grad=True))
        pretrain_eval_losses.append(averaged_task_loss.item())

# fine-tune on classification
fine_optim = torch.optim.Adam(list(eval_task.task_head.parameters()))

fine_losses = []
fine_eval_losses = []

indices = np.arange(data["X_train"].shape[0])
for j in range(fine_epochs):  # fine-tuning loop
    print("fine tuning epoch: ", j)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        batch = eval_task.pretreat(data["X_train"][indices[i - batch_size:i]])
        labels = torch.as_tensor(data["y_train"][indices[i - batch_size:i]], dtype=torch.long)

        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_losses.append(loss.item())

        fine_optim.zero_grad()  # backward pass the loss
        loss.backward()
        fine_optim.step()

    for i in range(batch_size, data["X_val"].shape[0], batch_size):
        batch = eval_task.pretreat(data["X_val"][i - batch_size:i])
        labels = torch.as_tensor(data["y_val"][i - batch_size:i], dtype=torch.long)

        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_eval_losses.append(loss.item())

plt.plot(pretrain_losses)
plt.plot(pretrain_eval_losses)
plt.show()
plt.plot(fine_losses)
plt.plot(fine_eval_losses)
plt.title("fine tuning on average all 4 ssl tasks")
plt.savefig("plots/multi_ssl_pretraining.png")
"""

# testing fine-tuning on classification
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32
pre_epochs = 5
fine_epochs = 1  # n-shot setting

task = Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead)  # defining model objects
# task = Tasks.Colorization3(latent_embedding_size, bbm.SimpleConvDecode)
# task = Tasks.Contrastive4(latent_embedding_size, bbm.SimpleTaskHead)
# task = Tasks.MaskedAutoEncoding(latent_embedding_size, bbm.SimpleConvDecode)
# task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)
eval_task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)
backbone = bbm.SimpleConvEncode(latent_embedding_size)

try:  # some tasks require data modification that requires "harmonization" with the original data format
    params = list(task.harmonization.parameters()) + list(backbone.parameters()) + list(task.task_head.parameters())
except AttributeError:
    params = list(backbone.parameters()) + list(task.task_head.parameters())
optim = torch.optim.Adam(params)

pretrain_losses = []
pretrain_eval_losses = []
indices = np.arange(data["X_train"].shape[0])
for j in range(pre_epochs):  # pretraining loop
    print("epoch: ", j + 1)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])  # forward pass the data
        labels = torch.as_tensor(data["y_train"][indices[i - batch_size:i]], dtype=torch.long)

        latent = backbone.forward(batch)
        try:
            loss = task.generate_loss(latent)
        except TypeError:
            loss = task.generate_loss(latent, labels)
        pretrain_losses.append(loss.item())

        optim.zero_grad()  # backward pass the loss
        loss.backward()
        optim.step()

    for i in range(batch_size, data["X_val"].shape[0], batch_size):
        batch = task.pretreat(data["X_val"][i - batch_size:i])  # forward pass the data
        labels = torch.as_tensor(data["y_val"][i - batch_size:i], dtype=torch.long)
        latent = backbone.forward(batch)
        try:
            loss = task.generate_loss(latent)
        except TypeError:
            loss = task.generate_loss(latent, labels)
        pretrain_eval_losses.append(loss.item())

# fine-tune on classification
fine_optim = torch.optim.Adam(list(eval_task.task_head.parameters()))

fine_losses = []
fine_eval_losses = []

indices = np.arange(data["X_train"].shape[0])
for j in range(fine_epochs):  # fine tuning loop
    print("fine tuning epoch: ", j)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        batch = eval_task.pretreat(data["X_train"][indices[i - batch_size:i]])
        labels = torch.as_tensor(data["y_train"][indices[i - batch_size:i]], dtype=torch.long)

        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_losses.append(loss.item())

        fine_optim.zero_grad()  # backward pass the loss
        loss.backward()
        fine_optim.step()

    for i in range(batch_size, data["X_val"].shape[0], batch_size):
        batch = eval_task.pretreat(data["X_val"][i - batch_size:i])
        labels = torch.as_tensor(data["y_val"][i - batch_size:i], dtype=torch.long)

        latent = backbone.forward(batch)  # forward pass the data
        loss = eval_task.generate_loss(latent, labels)
        fine_eval_losses.append(loss.item())

# plt.plot(pretrain_losses)
# plt.plot(pretrain_eval_losses)
# plt.show()
plt.plot(fine_losses)
plt.plot(fine_eval_losses)
plt.title("fine tuning on rotation initialization")
plt.savefig("plots/rotation_pretraining.png")
"""

# testing eval of a task on eval dataset
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32
epochs = 5

# task = Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead)  # defining model objects
# task = Tasks.Colorization3(latent_embedding_size, bbm.SimpleConvDecode)
# task = Tasks.Contrastive4(latent_embedding_size, bbm.SimpleTaskHead)
task = Tasks.MaskedAutoEncoding(latent_embedding_size, bbm.SimpleConvDecode)
backbone = bbm.SimpleConvEncode(latent_embedding_size)

try:
    params = list(task.harmonization.parameters()) + list(backbone.parameters()) + list(task.task_head.parameters())
except AttributeError:
    params = list(backbone.parameters()) + list(task.task_head.parameters())
optim = torch.optim.Adam(params)

eval_losses = []
losses = []

indices = np.arange(data["X_train"].shape[0])
for j in range(epochs):
    print("epoch: ", j + 1)
    np.random.shuffle(indices)
    for i in range(batch_size, data["X_train"].shape[0], batch_size):
        # print("loop: ", i)
        batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])  # forward pass the data
        latent = backbone.forward(batch)
        loss = task.generate_loss(latent)
        losses.append(loss.item())

        optim.zero_grad()  # backward pass the loss
        loss.backward()
        optim.step()

    for i in range(batch_size, data["X_val"].shape[0], batch_size):
        batch = task.pretreat(data["X_val"][i - batch_size:i])  # forward pass the data
        latent = backbone.forward(batch)
        loss = task.generate_loss(latent)
        eval_losses.append(loss.item())

plt.semilogy(losses)
plt.semilogy(eval_losses)
plt.show()
"""

# testing a full training loop with classification task
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32

task = Tasks.Cifar10Classification(latent_embedding_size, bbm.SimpleTaskHead)  # defining model objects
backbone = bbm.SimpleConvEncode(latent_embedding_size)
optim = torch.optim.Adam(list(backbone.parameters()) + list(task.task_head.parameters()))

losses = []
indices = np.arange(data["X_train"].shape[0])
np.random.shuffle(indices)
for i in range(batch_size, data["X_train"].shape[0], batch_size):  # one epoch
    # print("loop: ", i)
    batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])
    labels = data["y_train"][indices[i - batch_size:i]]

    latent = backbone.forward(torch.as_tensor(batch, dtype=torch.float))
    loss = task.generate_loss(latent, torch.as_tensor(labels, dtype=torch.long))
    losses.append(loss.item())

    optim.zero_grad()  # backward pass the loss
    loss.backward()
    optim.step()

batch = data["X_train"][:128]
latent = backbone.forward(torch.as_tensor(batch, dtype=torch.float))
predictions = task.task_head.forward(latent)
acc = torchmetrics.functional.accuracy(predictions, torch.as_tensor(data["y_train"][:128], dtype=torch.long),
                                       task="multiclass", num_classes=10)
print(acc)
plt.semilogy(losses)
plt.show()
"""

# testing a full training loop with mae task
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32

task = Tasks.MaskedAutoEncoding(latent_embedding_size, bbm.SimpleConvDecode)  # defining model objects
backbone = bbm.SimpleConvEncode(latent_embedding_size)
optim = torch.optim.Adam(list(task.harmonization.parameters()) +
                         list(backbone.parameters()) +
                         list(task.task_head.parameters()))
losses = []
indices = np.arange(data["X_train"].shape[0])
np.random.shuffle(indices)
for i in range(batch_size, data["X_train"].shape[0], batch_size):  # one epoch
    # print("loop: ", i)
    batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])  # forward pass the data
    latent = backbone.forward(batch)
    loss = task.generate_loss(latent)
    losses.append(loss.item())

    optim.zero_grad()  # backward pass the loss
    loss.backward()
    optim.step()

batch = task.pretreat(data["X_train"][:batch_size])
latent = backbone.forward(batch)
predictions = torch.nn.functional.sigmoid(task.task_head.forward(latent, batch.shape))
# visualize the prediction
fig, ax = plt.subplots(1, 3)
ax[0].imshow(data["X_train"][0].transpose(1, 2, 0) / 255)  # original data
ax[1].imshow(task.labels[0][0].permute(1, 2, 0).detach())  # applied mask
ax[2].imshow(predictions[0].permute(1, 2, 0).detach())  # recreated data
plt.show()

plt.semilogy(losses)
plt.show()
"""

# testing a full training loop with contrastive task
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32

task = Tasks.Contrastive4(latent_embedding_size, bbm.SimpleTaskHead)  # defining model objects
backbone = bbm.SimpleConvEncode(latent_embedding_size)
optim = torch.optim.Adam(list(backbone.parameters()) +
                         list(task.task_head.parameters()))
losses = []
indices = np.arange(data["X_train"].shape[0])
np.random.shuffle(indices)
for i in range(batch_size, data["X_train"].shape[0], batch_size):  # one epoch
    # print("loop: ", i)
    batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])  # forward pass the data
    latent = backbone.forward(batch)
    loss = task.generate_loss(latent)
    losses.append(loss.item())

    optim.zero_grad()  # backward pass the loss
    loss.backward()
    optim.step()

# idk what kind of metric or visualization to use for contrastive
plt.semilogy(losses)
plt.show()
"""

# testing a full training loop with colorization task
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32

task = Tasks.Colorization3(latent_embedding_size, bbm.SimpleConvDecode)  # defining model objects
backbone = bbm.SimpleConvEncode(latent_embedding_size)
optim = torch.optim.Adam(list(task.harmonization.parameters()) +
                         list(backbone.parameters()) +
                         list(task.task_head.parameters()))
losses = []
indices = np.arange(data["X_train"].shape[0])
np.random.shuffle(indices)
for i in range(batch_size, data["X_train"].shape[0], batch_size):  # one epoch
    # print("loop: ", i)
    batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])  # forward pass the data
    latent = backbone.forward(batch)
    loss = task.generate_loss(latent)
    losses.append(loss.item())  # note: losses are lower using colorization3

    optim.zero_grad()  # backward pass the loss
    loss.backward()
    optim.step()

batch = task.pretreat(data["X_train"][:batch_size])
latent = backbone.forward(batch)
predictions = task.task_head.forward(latent, (batch_size, task.desired_precision * 2,
                                              data["X_train"].shape[2], data["X_train"].shape[3]))
# visualize the prediction
predictions_a = torch.argmax(predictions[:, :task.desired_precision], dim=1, keepdim=True)
predictions_a = torch.special.logit(predictions_a / task.desired_precision) * 100  # 110 is used in the paper

predictions_b = torch.argmax(predictions[:, task.desired_precision:], dim=1, keepdim=True)
predictions_a = torch.special.logit(predictions_a / task.desired_precision) * 100

base_l = torch.as_tensor(rgb2lab(data["X_train"][:batch_size].transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)[:, 0])
base_l = base_l.unsqueeze(1)

reconstructed = torch.concatenate((base_l, predictions_a, predictions_b), dim=1).permute(0, 2, 3, 1)
reconstructed = lab2rgb(reconstructed)  # .transpose(0, 3, 1, 2)  # this triggers warnings about values not being in 
# range, someone should really do something bout that 

plt.imshow(reconstructed[0])
plt.show()

plt.semilogy(losses)
plt.show()
"""

# testing a full training loop with rotate task
"""
latent_embedding_size = 128  # hyper parameters
batch_size = 32

task = Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead)  # defining model objects
backbone = bbm.SimpleConvEncode(latent_embedding_size)
optim = torch.optim.Adam(list(backbone.parameters()) +
                         list(task.task_head.parameters()))
losses = []
indices = np.arange(data["X_train"].shape[0])
np.random.shuffle(indices)
for i in range(batch_size, data["X_train"].shape[0], batch_size):  # one epoch
    batch = task.pretreat(data["X_train"][indices[i - batch_size:i]])  # forward pass the data
    latent = backbone.forward(batch)
    loss = task.generate_loss(latent)
    losses.append(loss.item())

    optim.zero_grad()  # backward pass the loss
    loss.backward()
    optim.step()  # runs without error

batch = task.pretreat(data["X_train"][:32])
latent = backbone.forward(batch)
predictions = task.task_head.forward(latent)
acc = torchmetrics.functional.accuracy(predictions, torch.as_tensor(task.labels), task="multiclass", num_classes=4)
print(acc)
plt.semilogy(losses)
plt.show()
"""

# testing a training step with rotate task
"""
latent_embedding_size = 128  # hyper parameter

task = Tasks.Rotation(latent_embedding_size, bbm.SimpleTaskHead)  # defining model objects
backbone = bbm.SimpleConvEncode(latent_embedding_size)
optim = torch.optim.Adam(list(backbone.parameters()) +
                         list(task.task_head.parameters()))

batch = task.pretreat(data["X_train"][:32])  # forward pass the data
latent = backbone.forward(batch)
loss = task.generate_loss(latent)

optim.zero_grad()  # backward pass the loss
loss.backward()
optim.step()  # runs without error
"""

# testing color distortion
"""
batch, dist = da.color_distortions(data["X_train"][:32])
batch = np.array(batch)
print(dist)
print(batch.shape)
print(type(batch))
plt.imshow(batch[0].transpose(1, 2, 0) / 255)
plt.show()
"""

# testing gaussian blur
"""
batch, blur = da.gauss_blur(data["X_train"][:32])
batch = np.array(batch)
print(blur)
print(batch.shape)
print(type(batch))
plt.imshow(batch[0].transpose(1, 2, 0) / 255)
plt.show()
"""

# testing cropping
"""
batch, window = da.cropping((data["X_train"][:32]))
batch = np.array(batch)
print(window)
print(batch.shape)
print(type(batch))
plt.imshow(batch[0].transpose(1, 2, 0) / 255)
plt.show()
"""

# testing masking
"""
batch, mask = da.masking(data["X_train"][:32])
print(mask.shape)
print(batch.shape)
plt.imshow(batch[0].transpose(1, 2, 0) / 255)
plt.show()
plt.imshow(mask[0].transpose(1, 2, 0))
plt.show()
"""

# testing flip
"""
batch, lab = da.horizontal_flip(data["X_train"][:32])
print(lab)
print(batch.shape)
print(type(batch))
plt.imshow(batch[0].transpose(1, 2, 0) / 255)
plt.show()
"""

# testing rotate
"""
batch, lab = da.rotate(data["X_train"][:32])
print(lab)
print(batch.shape)
plt.imshow(batch[0].transpose(1, 2, 0) / 255)
plt.show()
"""
