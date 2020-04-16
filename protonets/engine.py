from tqdm import tqdm

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
				  # 获取模型的参数
            'model': kwargs['model'],
            # 获取数据的参数
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False
        }
			
			# 获取model的参数的与optimizer相关的配置值（如lr等）
        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])
			# 根据state参数，初始化学习率的衰减参数
        self.hooks['on_start'](state)
        # 当没有达到截止条件时
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            # 训练模型
            state['model'].train()
            # 用state参数重置评价指标并且进行optimizer回传
            self.hooks['on_start_epoch'](state)
            # 得到每个epoch的数据的条数
            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
