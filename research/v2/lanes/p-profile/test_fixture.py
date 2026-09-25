"""Small stdlib-only executable loop for wrapper control-flow tests, NOT training."""
import sys

COMPUTE_DTYPE = 'fixture'
RESULT = {'steps': [], 'syncs': 0, 'fetches': 0, 'cache_refreshes': 0}


class Loss:
    def __init__(self, model):
        self.model = model

    def backward(self):
        self.model.grad = 2 * self.model.value


class Model:
    value = 7.0
    grad = None

    def __call__(self, x, y):
        RESULT['steps'].append([self.value, x, y])
        return Loss(self)

    def refresh_weight_cache(self, dtype):
        RESULT['cache_refreshes'] += 1


def _muon_math(value):
    return value * 0.25


def _muon_step(model):
    model.value -= _muon_math(model.grad)


class Optimizer:
    def __init__(self, model):
        self.model = model

    def step(self, update_adamw):
        _muon_step(self.model)


def fetch():
    RESULT['fetches'] += 1
    return RESULT['fetches'], 0, None


def unscale_grads_(model):
    pass


def clip_and_sync_gradients(model):
    pass


def synchronize(device):
    RESULT['syncs'] += 1


def record_step_boundary(step):
    pass


def main():
    model = Model()
    optimizer = Optimizer(model)
    x, y, loader_state = fetch()
    step = 0
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 22
    device, adam_step_now = None, True
    while step < num_steps:
        synchronize(device)
        loss = model(x, y)
        loss.backward()
        x, y, loader_state = fetch()
        unscale_grads_(model)
        clip_and_sync_gradients(model)
        optimizer.step(update_adamw=adam_step_now)
        for m in [model]:
            m.refresh_weight_cache(COMPUTE_DTYPE)
        synchronize(device)
        record_step_boundary(step)
        step += 1
    RESULT['final'] = [model.value, model.grad]


if __name__ == '__main__':
    main()
