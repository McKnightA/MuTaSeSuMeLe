import Tasks
import BackBoneModel as bbm
import torch


# todo recreate MAML:
#  require distribution of tasks
#  require inner and outer learning rates
#  randomly initialize θ
#  while not done do
#   Sample batch of tasks Ti ∼ p(T )
#   for all Ti do
#    Evaluate ∇θLTi (fθ) with respect to K examples
#    Compute adapted parameters with gradient descent: θi = θ − α∇θLTi(fθ)
#   end for
#   Update θ ← θ − β∇θPTi∼p(T ) LTi(fθi)
#  end while


class MAML:

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

        params = list(self.backbone.parameters())  # this is a list of Parameter objects
        try:  # some tasks require data modification that requires "harmonization" with the original data format
            pass
            params += list(test_task.harmonization.parameters()) + list(test_task.task_head.parameters())
        except AttributeError:
            pass
            params += list(test_task.task_head.parameters())
        self.meta_optim = meta_optim(params)

    def inner_loop(self, task, data, grad_steps):
        """
        Evaluate ∇θLTi (fθ) with respect to K examples
        Compute adapted parameters with gradient descent: θi = θ − α∇θLTi(fθ)
        :return:
        """

        adapted_parameters = [list(self.backbone.parameters())]  # this is a list of Generators of Parameter objects
        try:  # some tasks require data modification that requires "harmonization" with the original data format
            pass
            adapted_parameters += [list(task.harmonization.parameters()), list(task.task_head.parameters())]
        except AttributeError:
            pass
            adapted_parameters += [list(task.task_head.parameters())]

        for step in range(grad_steps):
            pretreated = task.pretreat(data)
            embedding = self.backbone.forward(pretreated)  # todo beware the common backbone and updating the weights
            loss = task.generate_loss(embedding)

            # zero gradients
            for param_list in adapted_parameters:
                for p in param_list:
                    p.grad = None

            # calc gradients
            grads = []
            for param_list in adapted_parameters:
                # .grad expects the params to be a sequence of Tensor so maybe this needs to get to .data
                grads.append(torch.autograd.grad(loss, list(param_list), retain_graph=True))

            # update gradients
            for i in range(len(adapted_parameters)):
                for p, g in zip(adapted_parameters[i], grads[i]):  # Grad step
                    # todo (eventually) what happens if this is a different differentiable optimizer
                    p.data -= self.inner_lr * g

        return adapted_parameters

    def outer_loop(self, data_batch, inner_grad_steps):
        """

        :return:
        """
        data, labels = data_batch
        original_backbone = [param.clone().detach().requires_grad_(True) for param in list(self.backbone.parameters())]

        theta_i_prime = []

        # Sample batch of tasks
        for task in self.tasks:
            adapted = self.inner_loop(task, data, inner_grad_steps)
            for i in range(len(adapted)):
                adapted[i] = [param.clone().detach().requires_grad_(True) for param in adapted[i]]

            theta_i_prime.append(adapted)

            # need to return the backbone to the original weights so the next task can start from the start
            for backbone_param, original_param in zip(self.backbone.parameters(), original_backbone):
                backbone_param.data = original_param.data.clone().detach().requires_grad_(True)

        # by now the backbone still has its original Parameters even if the data isn't the same

        # Meta update
        self.meta_optim.zero_grad()
        meta_losses = []

        # for each set of adapted parameters compute the loss, then average those losses and take a grad step
        for adapted_params in theta_i_prime:
            # use the adapted parameters
            for backbone_param, adapted_param in zip(self.backbone.parameters(), adapted_params[0]):  # 0th is backbone
                backbone_param.data = adapted_param.data
                # could be a copy of the adapted data, but its not needed later so having it change is fine

            # compute the loss
            pretreated = self.test_task.pretreat(data)
            embedding = self.backbone.forward(pretreated)
            try:
                loss = self.test_task.generate_loss(embedding)
            except TypeError:
                loss = self.test_task.generate_loss(embedding, labels)

            meta_losses.append(loss)

        averaged_meta_loss = torch.mean(torch.tensor(meta_losses, requires_grad=True))
        averaged_meta_loss.backward(retain_graph=True)

        self.meta_optim.step()

        # where do I start and where do I end?

        # I start with some number of training tasks that all have different heads and maybe harmonization layers
        # I also start with a meta task that has its own head and maybe harmonization layers
        # all tasks share a common backbone

        # each training task ends with its layers updated by the inner optim (SGD) some number of times for each call
        # the meta task has its

        return averaged_meta_loss

