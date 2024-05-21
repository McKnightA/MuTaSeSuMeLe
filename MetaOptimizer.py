import torch


# TODO:
#  (now) debug


class MAML:
    """
    based on https://arxiv.org/abs/1703.03400
    """
    def __init__(self, training_tasks, test_task, backbone, inner_lr, meta_optim):
        """

        :param training_tasks:
        :param test_task:
        :param backbone:
        :param inner_lr:
        :param meta_optim:
        """
        self.tasks = training_tasks
        self.test_task = test_task
        self.backbone = backbone
        self.inner_lr = inner_lr

        # this is a list of Parameter objects
        params = list(self.backbone.parameters()) + list(test_task.parameters())

        self.meta_optim = meta_optim(params, lr=0.0001)

    def inner_loop(self, task, data_batch, grad_steps):
        """
        Evaluate ∇θLTi (fθ) with respect to K examples
        Compute adapted parameters with gradient descent: θi = θ − α∇θLTi(fθ)
        :return:
        """

        data, labels = data_batch

        # this is a list of Generators of Parameter objects
        adapted_parameters = [list(self.backbone.parameters())] + [list(task.parameters())]

        for step in range(grad_steps):
            try:
                loss = task.forward(data, self.backbone)
            except TypeError:
                loss = task.forward(data, labels, self.backbone)

            # zero gradients
            for param_list in adapted_parameters:
                for p in param_list:
                    p.grad = None

            # calc gradients
            grads = []
            for param_list in adapted_parameters:
                # .grad expects the params to be a sequence of Tensor
                grads.append(torch.autograd.grad(loss, list(param_list), retain_graph=True))

            # update gradients
            for i in range(len(adapted_parameters)):
                for p, g in zip(adapted_parameters[i], grads[i]):
                    # todo (eventually) what happens if this is a different differentiable optimizer
                    p.data -= self.inner_lr * g

        return adapted_parameters

    def outer_loop(self, data_batch, inner_grad_steps):
        """
        this should change to train_data_batch, meta_data_batch, inner_grad_steps
        :return:
        """
        data, labels = data_batch
        original_backbone = [param.clone().detach().requires_grad_(True) for param in list(self.backbone.parameters())]

        theta_i_prime = []

        # Sample batch of tasks
        for task in self.tasks:
            adapted = self.inner_loop(task, data_batch, inner_grad_steps)

            for i in range(len(adapted)):
                adapted[i] = [param.clone().detach().requires_grad_(True) for param in adapted[i]]
                # do I ever update the parameters of the task heads? yes, in the inner loop
                # why copy parameters for backbone and task head?
                # the meta update only needs the adapted parameters of the backbone
                # so copying the task head is unnecessary
                # todo remove unnecessary copying

            theta_i_prime.append(adapted)

            # need to return the backbone to the original weights so the next task can start from the start
            for backbone_param, original_param in zip(self.backbone.parameters(), original_backbone):
                backbone_param.data = original_param.data.clone().detach().requires_grad_(True)

        # by now the backbone has its original Parameters and inner loop task heads are updated for the data batch

        # Meta update
        self.meta_optim.zero_grad()
        meta_losses = []

        # for each set of adapted parameters compute the loss, then average those losses and take a grad step
        for adapted_params in theta_i_prime:
            # use the adapted parameters
            for backbone_param, adapted_param in zip(self.backbone.parameters(), adapted_params[0]):  # 0th is backbone
                backbone_param.data = adapted_param.data  # .clone().detach().requires_grad_(True)
                # could be a copy of the adapted data, but adapted is not needed later so having it change is fine

            # compute the loss
            try:
                loss = self.test_task.forward(data, self.backbone)
            except TypeError:
                loss = self.test_task.forward(data, labels, self.backbone)

            meta_losses.append(loss)

        # todo (eventually) this is a multi task operation
        #  so what if multi tasking methods where used instead?
        averaged_meta_loss = torch.mean(torch.tensor(meta_losses, requires_grad=True))
        averaged_meta_loss.backward(retain_graph=True)

        self.meta_optim.step()

        return averaged_meta_loss

