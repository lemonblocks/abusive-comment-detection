import time
import sys
import datetime

class Logger():
    "训练过程中打印并保存 metric"

    def __init__(self, n_epochs, batches_epoch):
        
        self.n_epochs = n_epochs    # 总epochs
        self.batches_epoch = batches_epoch  # 每个epochs中的bacth数
        self.epoch = 1              # 当前epoch
        self.batch = 1              # 当前batch
        self.prev_time = time.time()
        self.mean_period = 0
        self.metrics = {}
        self.metric_list = {}


    def log(self, metrics=None):
        """metrics: {"name1": value1, "name2": value2}
           value 为 size=1 的 torch Tensor
        """
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        # 打印当前/总Epoch与Batch数
        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        # 打印当前 Batch 中的所有 Metric
        for i, metric_name in enumerate(metrics.keys()):
            if metric_name not in self.metrics:
                self.metrics[metric_name] = metrics[metric_name].item()
            else:
                self.metrics[metric_name] += metrics[metric_name].item()

            if (i+1) == len(metrics.keys()):
                sys.stdout.write('%s: %.4f -- ' % (metric_name, self.metrics[metric_name]/self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (metric_name, self.metrics[metric_name]/self.batch))

        # 打印当前 Epoch 剩余时间
        batches_done = self.batches_epoch*(self.epoch - 1) + self.batch
        batches_left = self.batches_epoch*(self.n_epochs - self.epoch) + self.batches_epoch - self.batch 
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left*self.mean_period/batches_done)))

        # 保存当前 Epoch 的所有 Metric, 用于后续画图
        if (self.batch % self.batches_epoch) == 0:
            for metric_name, value in self.metrics.items():
                if metric_name not in self.metric_list:
                    self.metric_list[metric_name] = [value / self.batch]
                else:
                    self.metric_list[metric_name].append(value / self.batch)

                # Reset metrics for next epoch
                self.metrics[metric_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1
