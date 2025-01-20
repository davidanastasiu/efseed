import time
import os
import math
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import random
from ..utils.utils2 import (
    log_std_denorm_dataset,
    cos_date,
    sin_date,
    adjust_learning_rate,
)
from ..utils.metric import metric_rolling
from sklearn.metrics import mean_absolute_percentage_error
from .EFSEED_model import EncoderLSTM, DecoderLSTM
import zipfile

random.seed("a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EFSEED:

    def __init__(self, opt, dataset):
        #         super(EFSEED, self).__init__(opt)

        self.dataset = dataset
        self.opt = opt
        self.sensor_id = opt.stream_sensor
        self.dataloader = dataset.get_train_data_loader()
        self.trainX = dataset.get_trainX()
        self.val_data = np.array(dataset.get_val_points()).squeeze(1)
        self.data = dataset.get_data()
        self.sensor_data_norm = dataset.get_sensor_data_norm()
        self.sensor_data_norm_1 = dataset.get_sensor_data_norm1()
        self.R_norm_data = dataset.get_R_sensor_data_norm()
        self.mean = dataset.get_mean()
        self.std = dataset.get_std()
        self.month = dataset.get_month()
        self.day = dataset.get_day()
        self.hour = dataset.get_hour()

        self.train_days = opt.input_len
        self.predict_days = opt.output_len
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.is_watersheds = opt.watershed
        self.is_prob_feature = 1
        self.TrainEnd = opt.model
        self.os = opt.oversampling
        self.r_shift = opt.r_shift
        self.quantile = opt.quantile
        self.is_over_sampling = 1

        self.batchsize = opt.batchsize
        self.epochs = opt.epochs
        self.layer_dim = opt.layer
        self.predict_days = opt.output_len

        self.encoder = EncoderLSTM(self.opt).to(device)
        self.decoder = DecoderLSTM(self.opt).to(device)

        self.criterion = nn.MSELoss(reduction="sum")
        self.criterion1 = nn.HuberLoss(reduction="sum")
        self.criterion_KL = nn.KLDivLoss(reduction="sum")
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), self.opt.learning_rate)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), self.opt.learning_rate)

        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, "val")
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, "test")

        self.train_loss_list = []
        self.val_loss_list = []

    def get_train_loss_list(self):

        return self.train_loss_list

    def get_val_loss_list(self):

        return self.val_loss_list

    def std_denorm_dataset(self, predict_y0, pre_y):

        a2 = log_std_denorm_dataset(self.mean, self.std, predict_y0, pre_y)

        return a2

    
    def inference_test(self, x_test, y_input1):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
            y_input1 = torch.tensor(y_input1, dtype=torch.float32).to(device)

            h0 = torch.zeros(self.layer_dim, x_test.size(0), self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, x_test.size(0), self.hidden_dim).to(device)

            encoder_h, encoder_c = self.encoder(x_test, h0, c0)
            _, _, _, _, out4 = self.decoder(y_input1, x_test, encoder_h, encoder_c)
            y_predict = out4.cpu().numpy().flatten().reshape(1, -1)

        return y_predict

    def test_single(self, test_point):

        self.encoder.eval()
        self.decoder.eval()

        test_predict = np.zeros(self.predict_days * self.output_dim)

        # foot label of test_data
        point = self.trainX[self.trainX["datetime"] == test_point].index.values[0]
        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]
        test_point = point - start_num
        pre_gt = self.trainX[point - 1: point + self.opt.output_len - 1][
            "value"
        ].values.tolist()
        y = self.trainX[point: point + self.predict_days]["value"]

        b = test_point
        e = test_point + self.predict_days

        months = self.month[b:e]
        days = self.day[b:e]
        hours = self.hour[b:e]

        y2 = cos_date(months, days, hours) 
        y3 = sin_date(months, days, hours)

        y2 = np.reshape(y2, (-1, 1))
        y3 = np.reshape(y3, (-1, 1))        
        
        y_input1 = np.array([np.concatenate((y2, y3), 1)])

        # inference
        norm_data = self.sensor_data_norm
        if self.is_watersheds == 1:
            x_test = np.array(
                self.sensor_data_norm_1[test_point - self.train_days * 1: test_point],
                np.float32,
            ).reshape(self.train_days, -1)
        else:
            x_test = np.array(
                norm_data[test_point - self.train_days * 1: test_point], np.float32
            ).reshape(self.train_days, -1)

        x_test = [x_test]
        x_test = np.array(x_test)

        y_predict = self.inference_test(x_test, y_input1)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = y_predict.astype(float).tolist()
               
        pre_gt = self.data[test_point - 1: test_point]
        pre_gt = pre_gt[0]
        if pre_gt is None:
            print(pre_gt)

        test_predict = np.array(self.std_denorm_dataset(y_predict, pre_gt))
        test_predict = (test_predict + abs(test_predict)) / 2
        return test_predict, y

    def generate_single_val_rmse(self, min_RMSE=500):

        total = 0
        val_rmse_list = []
        val_pred_list = []
        val_pred_lists_print = []
        gt_mape_list = []
        val_mape_list = []
        val_points = self.val_data
        test_predict = np.zeros(self.predict_days * self.output_dim)

        non_flag = 0
        for i in range(len(val_points)):

            val_pred_list_print = []
            val_point = val_points[i]
            test_predict, ground_truth = self.test_single(val_point)
            rec_predict = test_predict

            for j in range(len(rec_predict)):
                temp = [val_point, j, rec_predict[j]]
                val_pred_list.append(temp)
                val_pred_list_print.append(rec_predict[j])

            val_pred_lists_print.append(val_pred_list_print)
            val_MSE = np.square(np.subtract(ground_truth, test_predict)).mean()
            val_RMSE = math.sqrt(val_MSE)
            val_rmse_list.append(val_RMSE)
            total += val_RMSE

            if np.isnan(ground_truth).any():
                print("val_point is: ", val_point)
                print("groud_truth:", ground_truth)
                non_flag = 1
            if np.isnan(test_predict).any():
                print("val_point is: ", val_point)
                print("there is non in test_predict:", test_predict)
                non_flag = 1
            gt_mape_list.extend(ground_truth)
            val_mape_list.extend(test_predict)


        new_min_RMSE = min_RMSE

        if total < min_RMSE:

            # save_model

            new_min_RMSE = total
            expr_dir = os.path.join(self.opt.outf, self.opt.name, "train")
            c_dir = os.getcwd()
            os.chdir(expr_dir)
            with zipfile.ZipFile(self.opt.name + ".zip", "w") as my_zip:
                with my_zip.open("EFSEED_encoder.pt", "w") as data_file:
                    torch.save(self.encoder.state_dict(), data_file)
            with zipfile.ZipFile(self.opt.name + ".zip", "a") as my_zip:
                with my_zip.open("EFSEED_decoder.pt", "w") as data_file:
                    torch.save(self.decoder.state_dict(), data_file)
            os.chdir(c_dir)
        print("val total RMSE: ", total)
        print("val min RMSE: ", new_min_RMSE)

        if non_flag == 0:
            mape = mean_absolute_percentage_error(
                np.array(gt_mape_list) + 1, np.array(val_mape_list) + 1
            )
        else:
            mape = 100

        return total, new_min_RMSE, mape

    def model_load(self):

        c_dir = os.getcwd()
        os.chdir(self.expr_dir)

        model1 = EncoderLSTM(self.opt).to(device)
        model2 = DecoderLSTM(self.opt).to(device)
        with zipfile.ZipFile(self.opt.name + ".zip", "r") as archive:
            with archive.open("EFSEED_encoder.pt", "r") as pt_file:
                model1.load_state_dict(torch.load(pt_file), strict=False)
                print("Importing the best EFSEED_encoder pt file:", pt_file)

        with zipfile.ZipFile(self.opt.name + ".zip", "r") as archive:
            with archive.open("EFSEED_decoder.pt", "r") as pt_file:
                model2.load_state_dict(torch.load(pt_file), strict=False)
                print("Importing the best EFSEED_decoder pt file:", pt_file)

        os.chdir(c_dir)
        self.encoder = model1
        self.decoder = model2

    def generate_test_rmse_mape(self):

        total = 0
        val_rmse_list = []
        val_pred_list = []
        val_pred_lists_print = []
        gt_mape_list = []
        val_mape_list = []
        val_points = self.test_data
        test_predict = np.zeros(self.predict_days * self.output_dim)

        non_flag = 0
        start = time.time()
        for i in range(len(val_points)):
            start = time.time()
            val_pred_list_print = []
            val_point = val_points[i]
            test_predict, ground_truth = self.test_single(val_point)
            test_predict = (test_predict + abs(test_predict)) / 2
            rec_predict = test_predict
            val_MSE = np.square(np.subtract(ground_truth, test_predict)).mean()
            val_RMSE = math.sqrt(val_MSE)
            val_rmse_list.append(val_RMSE)
            total += val_RMSE

            for j in range(len(rec_predict)):
                temp = [val_point, j, rec_predict[j]]
                val_pred_list.append(temp)
                val_pred_list_print.append(rec_predict[j])

            val_pred_lists_print.append(val_pred_list_print)
            gt_mape_list.extend(ground_truth)
            val_mape_list.extend(test_predict)
        end = time.time()
        print("Inferencing test points ", len(val_points), " use: ", end - start)

        if self.is_over_sampling == 1:
            OS = "_OS" + str(self.opt.oversampling)
        else:
            OS = "_OS-null"

        if self.is_watersheds == 1:
            if self.is_prob_feature == 0:
                watersheds = "shed"
            else:
                watersheds = "Shed-ProbFeature"
        elif self.is_prob_feature == 0:
            watersheds = "solo"
        else:
            watersheds = "ProbFeature"

        basic_path = self.test_dir + "/" + str(self.sensor_id) + OS

        if self.opt.save == 1:
            aa = pd.DataFrame(data=val_pred_lists_print)
            i_dir = (
                basic_path
                + "_"
                + watersheds
                + str(self.TrainEnd)
                + "_pred_lists_print.tsv"
            )
            aa.to_csv(i_dir, sep="\t")
            print("Inferencing result is saved in: ", i_dir)

        if non_flag == 0:
            mape = mean_absolute_percentage_error(
                np.array(gt_mape_list) + 1, np.array(val_mape_list) + 1
            )
        else:
            mape = 100

        return total, mape, val_pred_lists_print

    def inference(self):
        start = time.time()
        self.dataset.gen_test_data()  # generate test points file and test_data
        end = time.time()
        print("generate test points file and test_data: ", end - start)

        # refresh the related values
        self.test_data = np.array(self.dataset.get_test_points()).squeeze(1)  # read the test set
        self.data = self.dataset.get_data()
        self.sensor_data_norm = self.dataset.get_sensor_data_norm()
        self.sensor_data_norm_1 = self.dataset.get_sensor_data_norm1()
        self.R_norm_data = self.dataset.get_R_sensor_data_norm()
        self.mean = self.dataset.get_mean()
        self.std = self.dataset.get_std()
        self.month = self.dataset.get_month()
        self.day = self.dataset.get_day()
        self.hour = self.dataset.get_hour()
        rmse, mape, aa = self.generate_test_rmse_mape()  # inference on test set
        return aa

    def compute_metrics(self, aa):
        val_set = pd.read_csv(
            "./data_provider/datasets/test_timestamps_24avg.tsv", sep="\t"
        )
        val_points = val_set["Hold Out Start"]
        trainX = pd.read_csv(
            "./data_provider/datasets/" + self.opt.stream_sensor + ".csv", sep="\t"
        )
        trainX.columns = ["id", "datetime", "value"]
        count = 0
        for test_point in val_points:
            point = trainX[trainX["datetime"] == test_point].index.values[0]
            NN = np.isnan(trainX[point - self.train_days: point + self.predict_days]["value"]).any()
            if not NN:
                count += 1
        vals4 = aa
        # compute metrics
        all_GT = []
        all_EFSEED = []
        loop = 0
        ind = 0
        while loop < len(val_points):
            ii = val_points[loop]
            point = trainX[trainX["datetime"] == ii].index.values[0]
            x = trainX[point - self.train_days: point + self.predict_days]["value"].values.tolist()
            if np.isnan(np.array(x)).any():
                loop = loop + 1  # id for time list
                continue
            loop = loop + 1
            if ind >= count - count % 100:
                break
            ind += 1
            temp_vals4 = list(vals4[ind - 1])
            all_GT.extend(x[self.train_days:])
            all_EFSEED.extend(temp_vals4)
        metrics = metric_rolling(np.array(all_EFSEED), np.array(all_GT))
        return metrics

    def train(self):

        num_epochs = self.epochs
        early_stop = 0
        old_val_loss = 1000
        min_RMSE = 500000
        m = nn.Softmax(dim=1)

        for epoch in range(num_epochs):
            print_loss_total = 0  # Reset every epoch
            self.encoder.train()
            self.decoder.train()
            start = time.time()

            for i, batch in enumerate(self.dataloader):
                x_train = [TrainData for TrainData, _ in batch]
                x_train = np.array(x_train)
                y_train = [TrainLabel for _, TrainLabel in batch]
                y_train = np.array(y_train)

                y_train0 = y_train[:,:,0:1]
                y_train1 = y_train[:,:,1:3]

                x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
                y_train = torch.tensor(y_train0, dtype=torch.float32).to(device)
                decoder_input1 = torch.tensor(y_train1, dtype=torch.float32).to(device)

                q0 = torch.tensor([self.quantile / 100]).to(device)

                # compute ground segment label (Ensemble times=8, 32, 96, 288)
                s_l = int(self.predict_days / 36)  # expand times 4, 3, 3, 4*3*3=36
                y_train_s = y_train.view(y_train.size(0), s_l, int(self.predict_days / s_l)) 
                seg_label_g0 = torch.quantile(y_train_s, q0, dim=2)[0]

                y_train_s = y_train.view(y_train.size(0), 4 * s_l, int(self.predict_days / (4 * s_l)))
                seg_label_g1 = torch.quantile(y_train_s, q0, dim=2)[0]

                y_train_s = y_train.view(y_train.size(0), 12 * s_l, int(self.predict_days / (12 * s_l)))
                seg_label_g3 = torch.quantile(y_train_s, q0, dim=2)[0]

                seg_label_g4 = y_train.view(y_train.size(0), self.predict_days)

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                loss = 0

                h0 = torch.zeros(self.layer_dim, x_train.size(0), self.hidden_dim).to(device)
                c0 = torch.zeros(self.layer_dim, x_train.size(0), self.hidden_dim).to(device)
                # Forward pass
                encoder_h, encoder_c = self.encoder(x_train, h0, c0)
                out0, out1, out2, out3, out4 = self.decoder(decoder_input1, x_train, encoder_h, encoder_c)

                loss0 = self.criterion_KL(F.log_softmax(out0,dim=1), F.softmax(seg_label_g0,dim=1))  
                loss1 = self.criterion_KL(F.log_softmax(out1,dim=1), F.softmax(seg_label_g1,dim=1)) 
                loss3 = self.criterion_KL(F.log_softmax(out3,dim=1), F.softmax(seg_label_g3,dim=1))
                loss2 = self.criterion(out2[:,:16], seg_label_g4[:,:16]) 
                loss4 = self.criterion(out4, seg_label_g4) 

                l_w = max(-1 * np.exp(epoch/45) + 2, 0.1)
                loss = l_w *(200 *( loss0 + loss1 + loss3) + loss2) + loss4 

                loss.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
                print_loss_total += loss.item()

            self.encoder.eval()
            self.decoder.eval()

            val_loss, min_RMSE, mape = self.generate_single_val_rmse(min_RMSE)
            self.train_loss_list.append(print_loss_total)
            self.val_loss_list.append(val_loss)
            end = time.time()
            val_loss, min_RMSE, mape = self.generate_single_val_rmse(min_RMSE)
            self.train_loss_list.append(print_loss_total)
            self.val_loss_list.append(val_loss)
            end = time.time()
            print('-----------Epoch: {}. train_Loss>: {:.6f}. -------------'.format(epoch, print_loss_total)) 
            print('-----------Epoch: {}. val_Loss_rmse>: {:.6f}. ------------'.format(epoch, val_loss)) 
            print('-----------Epoch: {}. val_Loss_mape>: {:.6f}. -------------'.format(epoch, mape))
            print('-----------Epoch: {}. running time>: {:.6f}. -------------'.format(epoch, end-start))
            # early stop
            if val_loss > old_val_loss:
                early_stop += 1
            else:
                early_stop = 0
            if early_stop > 3:
                break
            old_val_loss = val_loss

            adjust_learning_rate(self.encoder_optimizer, epoch + 1, self.opt)
            adjust_learning_rate(self.decoder_optimizer, epoch + 1, self.opt)
