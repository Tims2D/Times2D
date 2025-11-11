import math
import numpy as np
import torch
import torch.nn as nn
import os
import time
import ptflops ###
import warnings
import matplotlib.pyplot as plt
from torch import optim
import psutil
import threading
from torch.optim import lr_scheduler
import torch.nn.functional as F
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
warnings.filterwarnings('ignore')
import subprocess  # Make sure to include this at the top of your file


class Exp_long_term_forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_long_term_forecasting, self).__init__(args)
        self.task_name =args.task_name
    def _build_model(self):
       
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

    def get_gpu_memory_usage(self):
        
        try:
        
            gpu_usage = os.popen('nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader').read()
            return int(gpu_usage.strip())  # Convert string to integer (MB)
        
        except Exception as e:
            
            return 0  # Or any other value indicating that the usage is not available
    
    
    def monitor_memory_usage(self, memory_usage_list, gpu_memory_usage_list, stop_event):
        while not stop_event.is_set():
            memory_usage = self.get_memory_usage()
            gpu_memory_usage = self.get_gpu_memory_usage()
            memory_usage_list.append(memory_usage)
            gpu_memory_usage_list.append(gpu_memory_usage)
            
            
            time.sleep(1)  # Adjust the sleep time as needed

    def get_gpu_info():
        try:
            # This runs the nvidia-smi command and captures the output
            gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        except FileNotFoundError:
            gpu_info = "nvidia-smi tool not found. It may not be installed or it's not in your PATH."
        return gpu_info
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

       # RAM_Usage meaurement
        process = psutil.Process() ###
        ram_before = process.memory_info().rss / 1024 ** 2 ###
        print(f"RAM before {ram_before}")
        
        time_list = []
        RAM_Usage_list = []
        memory_usage_list = []
        gpu_memory_usage_list = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self.monitor_memory_usage, args=(memory_usage_list, gpu_memory_usage_list, stop_event))
        monitor_thread.start()
        
        # >>> ADDED: training profiler accumulators
        data_time_sum = 0.0
        fwd_time_sum = 0.0
        bwd_time_sum = 0.0
        opt_time_sum = 0.0
        step_count = 0
        last_step_end_time = None  # to measure data loading (time until first tensors available)

        # >>> ADDED: reset CUDA peak stats
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                step_start_time = time.time()  # Start timing for this step
                # >>> ADDED: data loading time
                if last_step_end_time is None:
                    data_time = 0.0
                else:
                    data_time = step_start_time - last_step_end_time

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                #print('self.args.model =', self.args.model)
                
                # >>> ADDED: start forward timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_start = time.time()

                if self.args.model in ['Real_FITS', 'ModernTCN', 'SparseTSF','ConvLSTM']:
                    outputs = self.model(batch_x)
                
                elif self.args.model in ['HDMixer']:
                    outputs,PaEN_Loss = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # >>> ADDED: stop forward timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd_end = time.time()
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)
                else:
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                if self.args.model in ['HDMixer']:
                    mseloss = criterion(outputs, batch_y)
                    loss = mseloss+PaEN_Loss
                    train_loss.append(mseloss.item())
                else:   
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # >>> ADDED: start backward timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_bwd_start = time.time()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.cuda.empty_cache()
                    model_optim.step()

                # >>> ADDED: stop backward timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_opt_end = time.time()
                fwd_time_sum += (t_fwd_end - t_fwd_start)
                bwd_time_sum += (t_opt_end - t_bwd_start)
                data_time_sum += data_time
                step_count += 1
                last_step_end_time = t_opt_end

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # >>> ADDED: epoch-level wall-clock breakdown
            if step_count > 0:
                avg_data = data_time_sum / step_count
                avg_fwd  = fwd_time_sum / step_count
                avg_bwd  = bwd_time_sum / step_count
                total_step = avg_data + avg_fwd + avg_bwd
                if total_step <= 0: total_step = 1e-9
                print("---- Wall-clock (this epoch, per-step avg) ----")
                print(f"Data loader:        {avg_data:.6f} s  ({100.0*avg_data/total_step:5.1f}%)")
                print(f"Forward pass:       {avg_fwd:.6f} s  ({100.0*avg_fwd/total_step:5.1f}%)")
                print(f"Backward+Optimizer: {avg_bwd:.6f} s  ({100.0*avg_bwd/total_step:5.1f}%)")
                print("------------------------------------------------")

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                t1 = (time.time() - epoch_time)
                time_list.append(t1)
                average_time = sum(time_list) / len(time_list)
                ram_after = process.memory_info().rss / 1024 ** 2
                RAM_usage = ram_after - ram_before 
                RAM_Usage_list.append(RAM_usage)
                RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)
                stop_event.set()
                monitor_thread.join()
                average_memory_usage = sum(memory_usage_list) / len(memory_usage_list)
                average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        
        t1 = (time.time() - epoch_time)
        time_list.append(t1)
        average_time = sum(time_list) / len(time_list)

        ram_after = process.memory_info().rss / 1024 ** 2
        RAM_usage = ram_after - ram_before
        RAM_Usage_list.append(RAM_usage)
        RAM_usage = sum(RAM_Usage_list) / len(RAM_Usage_list)
            
        print(f"epoch is {epoch}")
        if epoch == (self.args.train_epochs - 1):
            stop_event.set()
            monitor_thread.join()

        average_memory_usage = sum(memory_usage_list) / len(memory_usage_list)
        average_gpu_memory_usage = sum(gpu_memory_usage_list) / len(gpu_memory_usage_list)

        print('_______________________________________GPU Information_____________________________________')
        print(Exp_long_term_forecasting.get_gpu_info())
        
        print('_______________________________________Efficiency and Running Time_____________________________________')
        print(f"| {'Metric':<40} | {'Value':>20} |")
        print("--------------------------------------------------------------------------------------------------------")
        print(f"| {'Average training time per epoch':<40} | {average_time:>20.4f} seconds |")
        print(f"| {'RAM before':<40} | {ram_before:>20.2f} MB |")
        print(f"| {'RAM after':<40} | {ram_after:>20.2f} MB |")
        print(f"| {'RAM usage (After -Before) per epoch':<40} | {RAM_usage:>20.2f} MB |")
        print(f"| {'Average memory usage':<40} | {average_memory_usage:>20.2f} MB |")
        print(f"| {'Average GPU memory usage':<40} | {average_gpu_memory_usage:>20.2f} MB |")
        print()
        
        print('_______________________________________CPU Information_____________________________________')
        print(f"| {'Metric':<40} | {'Value':>20} |")
        print("--------------------------------------------------------------------------------------------------------")
        physical_cores = psutil.cpu_count(logical=False)
        total_cores = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"| {'Physical cores':<40} | {physical_cores:>20} |")
        print(f"| {'Total cores':<40} | {total_cores:>20} |")
        print(f"| {'Total CPU Usage':<40} | {cpu_usage:>20.2f} % |")
        print("--------------------------------------------------------------------------------------------------------")

        # >>> ADDED: GPU peak/reserved memory + overall wall-clock summary
        try:
            print('_______________________________________GPU Memory (Peak/Reserved)_____________________________________')
            if torch.cuda.is_available():
                device_idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_idx)
                total_vram = getattr(props, 'total_memory', 0) / (1024**2)
                peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                try:
                    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
                except Exception:
                    peak_reserved = float('nan')
                try:
                    curr_reserved = torch.cuda.memory_reserved() / (1024**2)
                except Exception:
                    curr_reserved = float('nan')
                print(f"| {'GPU total VRAM':<40} | {total_vram:>20.2f} MB |")
                print(f"| {'Peak allocated (since reset)':<40} | {peak_alloc:>20.2f} MB |")
                print(f"| {'Peak reserved (since reset)':<40} | {peak_reserved:>20.2f} MB |")
                print(f"| {'Current reserved':<40} | {curr_reserved:>20.2f} MB |")
            else:
                print("CUDA not available.")
            print("--------------------------------------------------------------------------------------------------------")
        except Exception:
            pass

        try:
            if step_count > 0:
                overall_avg_data = data_time_sum / step_count
                overall_avg_fwd  = fwd_time_sum / step_count
                overall_avg_bwd  = bwd_time_sum / step_count
                total_step = overall_avg_data + overall_avg_fwd + overall_avg_bwd
                if total_step <= 0: total_step = 1e-9
                print('_______________________________________Overall Wall-Clock Breakdown (Per-step Avg)_____________________')
                print(f"| {'Data loader':<40} | {overall_avg_data:>10.6f} s | {100.0*overall_avg_data/total_step:>7.2f}% |")
                print(f"| {'Forward pass':<40} | {overall_avg_fwd:>10.6f} s | {100.0*overall_avg_fwd/total_step:>7.2f}% |")
                print(f"| {'Backward+Optimizer':<40} | {overall_avg_bwd:>10.6f} s | {100.0*overall_avg_bwd/total_step:>7.2f}% |")
                print("--------------------------------------------------------------------------------------------------------")
        except Exception:
            pass
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.model in ['RNN', 'GRU','LSTM', 'BiLSTM', 'ResLSTM', 'Real_FITS', 'ModernTCN', 'SparseTSF','ConvLSTM']:                    
                    outputs = self.model(batch_x)
                elif self.args.model in ['HDMixer']:
                    outputs,PaEN_Loss = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)
                elif self.task_name == 'Univariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds, trues, inputx = [], [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # >>> ADDED: inference profiler accumulators
        inf_samples = 0
        inf_batches = 0
        inf_time_sum = 0.0
        inf_latencies = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # >>> ADDED: start inference timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inf_start = time.time()

                if self.args.model in ['RNN', 'GRU', 'LSTM', 'BiLSTM', 'ResLSTM','Real_FITS', 'ModernTCN', 'SparseTSF','ConvLSTM']:
                    outputs = self.model(batch_x)
                elif self.args.model in ['HDMixer']:
                    outputs,PaEN_Loss = self.model(batch_x)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # >>> ADDED: stop inference timer
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_inf_end = time.time()
                batch_latency = t_inf_end - t_inf_start
                inf_time_sum += batch_latency
                inf_latencies.append(batch_latency)
                inf_batches += 1
                try:
                    inf_samples += int(batch_x.size(0))
                except Exception:
                    pass

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.task_name == 'Multivariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, -1].unsqueeze(-1).to(self.device)
                elif self.task_name == 'Univariate_forecasting':
                     batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                     batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 1 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n\n')
        f.close()
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)

        # >>> ADDED: Inference-time reporting
        try:
            if inf_batches > 0:
                avg_latency = inf_time_sum / inf_batches
            else:
                avg_latency = float('nan')
            if inf_time_sum > 0:
                throughput = inf_samples / inf_time_sum
            else:
                throughput = float('nan')
            if len(inf_latencies) > 0:
                lat_sorted = sorted(inf_latencies)
                def _pct(arr, p):
                    k = int(round((p/100.0)*(len(arr)-1)))
                    return arr[k]
                p50 = _pct(lat_sorted, 50)
                p95 = _pct(lat_sorted, 95)
            else:
                p50 = float('nan')
                p95 = float('nan')
            print('_______________________________________Inference-Time Results__________________________________________')
            print(f"| {'Avg latency per batch (forward only)':<55} | {avg_latency:>12.6f} s |")
            print(f"| {'p50 latency per batch':<55} | {p50:>12.6f} s |")
            print(f"| {'p95 latency per batch':<55} | {p95:>12.6f} s |")
            print(f"| {'Throughput (samples/sec)':<55} | {throughput:>12.2f} |")
            print("--------------------------------------------------------------------------------------------------------")
        except Exception:
            pass
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'PDF' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'PDF' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)
        return
