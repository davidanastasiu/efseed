import numpy as np
from sklearn.metrics import mean_absolute_percentage_error


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    return mean_absolute_percentage_error(np.array(true) + 1, np.array(pred) + 1)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(model, pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape  # , mspe


def metric_g(name, pre, gt, predict_days=288):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre) / predict_days)
    mae_all = []  # unused?
    mse_all = []  # unused?
    rmse_all = []
    mape_all = []
    l2 = []
    l3 = []
    lll = []
    for i in range(ll):
        mae, mse, rmse, mape = metric(
            name,
            pre[i * predict_days: (i + 1) * predict_days],
            gt[i * predict_days: (i + 1) * predict_days],
        )
        rmse_all.append(rmse)
        mape_all.append(mape)
    l2.append(np.around(np.mean(np.array(rmse_all)), 2))
    l3.append(np.around(np.mean(np.array(mape_all)), 3))
    lll.append(l2)
    lll.append(l3)
    return lll


def metric_rolling(pre, gt, predict_days=288):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre) / predict_days)
    rmse_all1 = []
    mape_all1 = []
    rmse_all2 = []
    mape_all2 = []
    for i in range(ll):
        _, _, rmse1, mape1 = metric(
            "EFSEED",
            pre[i * predict_days: (i * predict_days + predict_days)],
            gt[i * predict_days: (i * predict_days + predict_days)],
        )
        _, _, rmse2, mape2 = metric(
            "EFSEED",
            pre[i * predict_days: (i * predict_days + 16)],
            gt[i * predict_days: (i * predict_days + 16)],
        )
        rmse_all1.append(rmse1)
        mape_all1.append(mape1)
        rmse_all2.append(rmse2)
        mape_all2.append(mape2)
    rmse1 = np.around(np.mean(np.array(rmse_all1)), 2)
    mape1 = np.around(np.mean(np.array(mape_all1)), 3)
    print("For rolling prediction: 3 days")
    print("RMSE: ", rmse1)
    print("MAPE: ", mape1)
    rmse2 = np.around(np.mean(np.array(rmse_all2)), 2)
    mape2 = np.around(np.mean(np.array(mape_all2)), 3)
    print("For rolling prediction: 4 hours")
    print("RMSE: ", rmse2)
    print("MAPE: ", mape2)
    return rmse1, mape1, rmse2, mape2
