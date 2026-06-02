
"""
Core utility functions and model definitions for wheat yield prediction.
This module implements:
    - Feature engineering for natural and socio-economic datasets
    - Data filtering and reshaping utilities
    - Trend decomposition using LOWESS
    - Data normalization
    - Deep learning model architectures
    - Model training pipeline including the Fit-Generalization Adaptive Attention (FGAA) algorithm
    - Model interpretation using Integrated Gradients and GradientShap
Dependencies:
    PyTorch, NumPy, Pandas, scikit-learn, statsmodels, Captum
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from d2l import torch as d2l
from captum.attr import IntegratedGradients, GradientShap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.nonparametric.smoothers_lowess import lowess
import time

### Construct environmental features.
def natural_feature_builder(natural_dataset):
    natural_featured = natural_dataset[['temperature_2m', 'dewpoint_temperature_2m', 'surface_solar_radiation_downwards_sum', 
                                        'surface_pressure', 'total_precipitation_sum']].copy()
    soil_temperature = natural_dataset[['soil_temperature_level_1','soil_temperature_level_2','soil_temperature_level_3']].mean(1)
    volumetric_soil_water = natural_dataset[['volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
                                             'volumetric_soil_water_layer_3']].mean(1)
    wind_speed_10m = (natural_dataset['u_component_of_wind_10m']**2 + natural_dataset['v_component_of_wind_10m']**2)**(1/2)
    natural_featured['soil_temperature'] = soil_temperature
    natural_featured['volumetric_soil_water'] = volumetric_soil_water
    natural_featured['wind_speed_10m'] = wind_speed_10m
    natural_feature = natural_featured[['surface_solar_radiation_downwards_sum', 'temperature_2m', 'soil_temperature', 
    'dewpoint_temperature_2m', 'volumetric_soil_water','total_precipitation_sum', 'surface_pressure', 'wind_speed_10m']]
    return natural_feature

### Remove samples with missing values and filter counties with insufficient observations.
def filter_(yield_data, social_data, natural_data, county_index):
    yield_flatten, num_year = yield_data.reshape(-1), yield_data.shape[1]
    social_flatten = social_data.reshape(yield_flatten.size(0),-1)
    natural_flatten = natural_data.reshape(yield_flatten.size(0),-1)
    index_nan = []
    for i in range(yield_flatten.size(0)):
        if torch.isnan(yield_flatten[i]) or torch.isnan(social_flatten[i,:]).any() or torch.isnan(natural_flatten[i,:]).any():
            index_nan.append(i)
    yield_flatten[index_nan] = np.nan
    yield_data = yield_flatten.reshape(-1,num_year)
    social_data = social_flatten.reshape(yield_data.size(0),num_year,-1)
    natural_data = natural_flatten.reshape(yield_data.size(0),num_year,-1)
    yield_data_csv = pd.DataFrame(yield_data.numpy())
    yield_notnan_bool = ~yield_data_csv.T.isna()
    county_notnan = np.arange(yield_data.size(0))[yield_notnan_bool.sum()>=10]
    yield_notnan, social_notnan, natural_notnan = yield_data[county_notnan], social_data[county_notnan], natural_data[county_notnan]
    if county_index is not None:
        county_notnan_index = county_index[county_notnan]
        return yield_notnan, social_notnan, natural_notnan, county_notnan_index
    else:
        return yield_notnan, social_notnan, natural_notnan

### Construct socioeconomic features.
def social_feature_builder(social_flatten):
    social_feature = torch.zeros([social_flatten.shape[0],8])
    social_feature[:,0] = social_flatten[:,1]/social_flatten[:,0] #Population Size
    social_feature[:,1] = social_flatten[:,8]/social_flatten[:,1] #Health-care Resource
    social_feature[:,2] = social_flatten[:,7]/social_flatten[:,1] #Pupil Proportion
    social_feature[:,3] = social_flatten[:,6]/social_flatten[:,1] #Teenager Proportion
    social_feature[:,4] = social_flatten[:,2]/social_flatten[:,1] #Fiscal Revenue
    social_feature[:,5] = social_flatten[:,3]/social_flatten[:,1] #Fiscal Expenditure
    social_feature[:,6] = social_flatten[:,4]/social_flatten[:,1] #Residential Savings
    social_feature[:,7] = social_flatten[:,5]/social_flatten[:,1] #Institutional Loans
    return social_feature

### Convert county-year matrices into sample-level tensors.
### Each sample corresponds to one county-year observation.
def flatten(yield_data, social_data, natural_data, county_index):
    yield_flat, num_year = yield_data.reshape(-1), yield_data.shape[1]
    social_flat, natural_flat = social_data.reshape(-1,9), natural_data.reshape(-1,72)
    index_notnan = np.arange(yield_flat.size(0))[~torch.isnan(yield_flat)]
    yield_flat = yield_flat[index_notnan].reshape(-1,1)
    social_flat = social_feature_builder(social_flat[index_notnan,:])
    natural_flat = natural_flat[index_notnan,:]
    if county_index is not None:
        county_year_index = np.concatenate([np.repeat(county_index,num_year,0), np.tile(np.arange(num_year), 
        len(county_index)).reshape(-1,1)], 1)
        county_year_index = county_year_index[index_notnan,:]
        return yield_flat, social_flat, natural_flat, county_year_index
    else:
        return yield_flat, social_flat, natural_flat

### Decompose each county-level time series into observed value, long-term trend, and residual component.
### LOWESS smoothing is used to estimate the trend component.
def resolve(yield_data, social_data, natural_data, county_index=None, frac=0.3):
    def decompose_trend(data, frac, data_type):
        if data_type == 'social':
            data = social_feature_builder(data.reshape(-1,9)).reshape(data.shape[0], data.shape[1], 8)
        if data_type in ['social', 'natural']:
            data = data.permute(0,2,1).reshape(-1,22)
        trend, years = np.full_like(data, np.nan, dtype=np.float64), np.arange(22)
        for i in range(data.shape[0]):
            data_county = data[i, :].numpy()
            mask_nan = np.isnan(data_county)
            valid_idx = np.where(~mask_nan)[0]
            start, end = valid_idx[0], valid_idx[-1]
            data_county, year = data_county[start:end+1], years[start:end+1]
            data_county = pd.Series(data_county).interpolate(limit_direction='both').to_numpy()
            data_trend = lowess(data_county, year, frac=frac, return_sorted=False)
            data_trend_valid = np.full(end-start+1, np.nan)
            data_trend_valid[~mask_nan[start:end+1]] = data_trend[~mask_nan[start:end+1]]
            trend[i, start:end+1] = data_trend_valid
        trend = torch.from_numpy(trend).float()
        resid = data - trend
        if data_type in ['social', 'natural']:
            data, trend, resid = data.reshape(574,-1,22), trend.reshape(574,-1,22), resid.reshape(574,-1,22)
            data, trend, resid = data.permute(0,2,1), trend.float().permute(0,2,1), resid.float().permute(0,2,1)
        if data_type == 'yield':
            data, trend, resid = data.reshape(-1), trend.reshape(-1), resid.reshape(-1)
        elif data_type == 'social':
            data, trend, resid = data.reshape(-1,8), trend.reshape(-1,8), resid.reshape(-1,8)
        else:
            data, trend, resid = data.reshape(-1,72), trend.reshape(-1,72), resid.reshape(-1,72)
        return data, trend, resid
    yield_obsvd, yield_trend, yield_resid = decompose_trend(yield_data, frac, 'yield')
    social_obsvd, social_trend, social_resid = decompose_trend(social_data, frac, 'social')
    natural_obsvd, natural_trend, natural_resid = decompose_trend(natural_data, frac, 'natural')
    index_notnan = np.arange(yield_trend.size(0))[~torch.isnan(yield_trend)]
    yield_obsvd = yield_obsvd[index_notnan].reshape(-1,1)
    yield_trend, yield_resid = yield_trend[index_notnan].reshape(-1,1), yield_resid[index_notnan].reshape(-1,1)
    social_obsvd, natural_obsvd = social_obsvd[index_notnan], natural_obsvd[index_notnan]
    social_trend, natural_trend = social_trend[index_notnan], natural_trend[index_notnan]
    social_resid, natural_resid = social_resid[index_notnan], natural_resid[index_notnan]
    yield_tear = torch.concat([yield_obsvd, yield_trend, yield_resid], 1)
    social_tear, natural_tear = [social_obsvd, social_trend, social_resid], [natural_obsvd, natural_trend, natural_resid]
    if county_index is not None:
        county_year_index = np.concatenate([np.repeat(county_index,22,0), np.tile(np.arange(22), len(county_index)).reshape(-1,1)], 1)
        county_year_index = county_year_index[index_notnan,:]
        return yield_tear, social_tear, natural_tear, county_year_index
    else:
        return yield_tear, social_tear, natural_tear

def scaler(data_train, data_valid, data_test):
    scaler = StandardScaler()
    scaler.fit(data_train)
    train_scaled, valid_scaled, test_scaled = scaler.transform(data_train), scaler.transform(data_valid), scaler.transform(data_test)
    train_scaled = torch.from_numpy(train_scaled).float()
    valid_scaled = torch.from_numpy(valid_scaled).float()
    test_scaled = torch.from_numpy(test_scaled).float()
    return train_scaled, valid_scaled, test_scaled

### Neural network architecture for yield prediction.
### Supports multiple model types including RNN, LSTM, GRU, CNN, attention-based models.
class Model(nn.Module):
    def __init__(self, model_type, data_type, num_instance_identity, require_identity_output=False):
        super().__init__()
        input_size = {'baseline':72, 'addition':80}.get(data_type)
        self.layer = {
            'rnn': nn.RNN(8, 72, num_layers=2, batch_first=True),
            'lst': nn.LSTM(8, 72, num_layers=2, batch_first=True),
            'gru': nn.GRU(8, 72, num_layers=2, batch_first=True),
            'cnn': nn.Conv1d(8, 8, kernel_size=3, padding=1),
            'att': nn.Linear(1, 8)
        }.get(model_type)
        self.layer_att = nn.MultiheadAttention(8, 2)
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, num_instance_identity), nn.ReLU()
        )
        self.dense = nn.Linear(num_instance_identity, 1)
        self.model_type, self.data_type, self.require_identity_output = model_type, data_type, require_identity_output
    def forward(self, input_data, test_data=False):
        if self.model_type == 'dnn':
            net_input = input_data
        elif self.model_type == 'att':
            net_input = input_data.unsqueeze(-1)
            net_input = self.layer(net_input)
            net_input = self.layer_att(net_input, net_input, net_input)[0].mean(-1)
            net_input = input_data + net_input
        else:
            net_input = input_data[:, 8:] if self.data_type == 'addition' else input_data
            net_input = net_input.reshape(-1, 9, 8)
            if self.model_type in ['rnn', 'lst', 'gru']:
                net_input = self.layer(net_input)[0].mean(dim=1)
            elif self.model_type == 'cnn':
                net_input = self.layer(net_input.permute(0,2,1)).permute(0,2,1).reshape(-1,72)
            if self.data_type == 'addition':
                net_input = torch.cat([input_data[:,:8], net_input], -1)
        mid_output = self.network(net_input)
        last_output = self.dense(mid_output)
        if self.require_identity_output is True:
            return last_output, mid_output
        else:
            return last_output

def attention(instance_identity):
    distance = instance_identity.size(-1)
    instance_scores = torch.matmul(instance_identity, instance_identity.transpose(0,1))/torch.sqrt(torch.tensor(distance))
    instance_scores = instance_scores.mean(1)
    return instance_scores

def presenter(model, data_iter, yield_observed):
    instance_criterion, instance_identity, yield_hat, yield_mse = nn.MSELoss(reduction='none'), [], [], []
    for yield_batch, input_batch in data_iter:
        yield_batch_hat, instance_identity_batch = model(input_batch)
        yield_batch_mse = instance_criterion(yield_batch_hat, yield_batch)
        instance_identity.append(instance_identity_batch.detach())
        yield_hat.append(yield_batch_hat.detach().cpu())
        yield_mse.append(yield_batch_mse.detach().cpu())
    instance_identity = torch.concat(instance_identity)
    yield_hat, yield_mse = np.concatenate(yield_hat), np.concatenate(yield_mse)
    num_output = next(iter(data_iter))[0].shape[1]
    r2 = r2_score(yield_observed.cpu(), yield_hat)
    mse = mean_squared_error(yield_observed.cpu(), yield_hat)
    mae = mean_absolute_error(yield_observed.cpu(), yield_hat)
    instance_scores = attention(instance_identity)
    return r2, mse, mae, yield_mse, instance_scores.cpu().numpy().reshape(-1,1)

### Compute feature attribution using Integrated Gradients and GradientShap.
def explainer(model_type, data_type, num_instance_identity, model_dict, data_to_explain, explainer_type):
    model_to_explain = Model(model_type, data_type, num_instance_identity)
    model_to_explain.load_state_dict(model_dict)
    model_to_explain.eval()
    if explainer_type == 'ig':
        model_for_explain = IntegratedGradients(model_to_explain)
    elif explainer_type == 'gs':
        model_for_explain = GradientShap(model_to_explain)
    feature_importance = []
    for yield_batch, input_batch in data_to_explain:
        feature_importance_batch = model_for_explain.attribute(input_batch.cpu(), baselines=torch.zeros(input_batch.size()))
        feature_importance.append(feature_importance_batch)
    feature_importance = torch.concat(feature_importance)
    return feature_importance.numpy()


### Train the yield prediction model using a two-stage strategy.
def predictor(model_type, data_type, num_instance_identity, require_explanation,yield_train, social_train, natural_train, yield_valid,
social_valid, natural_valid,batch_size, learning_rate, max_epoch, patience, train_breaker, yield_test, social_test, natural_test):
    model = Model(model_type, data_type, num_instance_identity, True).cuda()
    if data_type == 'baseline':
        input_train, input_valid, input_test = natural_train, natural_valid, natural_test
    elif data_type == 'addition':
        input_train, input_valid = torch.cat([social_train, natural_train], -1), torch.cat([social_valid, natural_valid], -1)
        input_test = torch.cat([social_test, natural_test], -1)
    yield_train, yield_valid, yield_test = yield_train.cuda(), yield_valid.cuda(), yield_test.cuda()
    input_train, input_valid, input_test = input_train.cuda(), input_valid.cuda(), input_test.cuda()
    train_iter = d2l.load_array((yield_train, input_train), batch_size, False)
    valid_iter = d2l.load_array((yield_valid, input_valid), batch_size, False)
    test_iter = d2l.load_array((yield_test, input_test), batch_size, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion, instance_criterion, metric = nn.MSELoss(), nn.MSELoss(reduction='none'), d2l.Accumulator(2)
    patience, best_loss, early_stopping_counter = patience, None, 0
    train_accuracy_gather, test_accuracy_gather, train_instance_gather, test_instance_gather, feature_importance = [], [], [], [], []
    test_r2, test_mse, test_mae, yield_test_hat, yield_test_mse, test_instance_scores = [], [], [], [], [], []

### Stage 1: Standard training with MSE loss.
    for epoch in range(max_epoch):
        model.train()
        for yield_train_batch, input_train_batch in train_iter:
            optimizer.zero_grad()
            yield_train_batch_hat, instance_identity_batch = model(input_train_batch)
            loss_batch = criterion(yield_train_batch_hat, yield_train_batch)
            loss_batch.backward()
            optimizer.step()
        model.eval()
        if epoch == 0 and train_breaker is True:
            train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
            test_r2, test_mse, test_mae, yield_test_mse, test_instance_scores = presenter(model, test_iter, yield_test)
            train_accuracy_gather.append([train_r2, train_mse, train_mae])
            test_accuracy_gather.append([test_r2, test_mse, test_mae])
            train_instance_gather.append([yield_train_mse, train_instance_scores])
            test_instance_gather.append([yield_test_mse, test_instance_scores])
        for yield_valid_batch, input_valid_batch in valid_iter:
            yield_valid_batch_hat, instance_identity_batch = model(input_valid_batch)
            loss_batch = criterion(yield_valid_batch_hat, yield_valid_batch)
            metric.add(loss_batch.detach(), 1)
        valid_metric = metric[0] / metric[1]
        metric.reset()
        if best_loss is None or best_loss > valid_metric:
            best_loss = valid_metric
            early_stopping_counter = 0
            model_dict = model.state_dict()
        else:
            early_stopping_counter +=1
            if early_stopping_counter >= patience:
                best_loss = None
                break
    model.load_state_dict(model_dict)
    train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
    test_r2, test_mse, test_mae, yield_test_mse, test_instance_scores = presenter(model, test_iter, yield_test)
    if require_explanation is True:
        for i in ['ig', 'gs']:
            test_feature_importance = explainer(model_type, data_type, num_instance_identity, model_dict, test_iter, i)
            feature_importance.append(test_feature_importance)
    train_accuracy_gather.append([train_r2, train_mse, train_mae])
    test_accuracy_gather.append([test_r2, test_mse, test_mae])
    train_instance_gather.append([yield_train_mse, train_instance_scores])
    test_instance_gather.append([yield_test_mse, test_instance_scores])

# Stage 2: FGAA fine-tuning, where samples with higher prediction difficulty receive larger optimization weights.
    train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
    instance_scores = torch.from_numpy(train_instance_scores.reshape(-1)).float().cuda()
    train_iter_finetune = d2l.load_array((yield_train, input_train, instance_scores), batch_size, False)

    for epoch in range(max_epoch):
        model.train()
        for yield_train_batch, input_train_batch, instance_scores_batch in train_iter_finetune:
            optimizer.zero_grad()
            yield_train_batch_hat, instance_identity_batch = model(input_train_batch)
            yield_train_batch_mse = instance_criterion(yield_train_batch_hat, yield_train_batch).mean(1)
            instance_weight_batch = instance_scores_batch / instance_scores_batch.sum()
            instance_loss_batch = torch.mm(instance_weight_batch.unsqueeze(0), yield_train_batch_mse.unsqueeze(1))
            instance_loss_batch.backward()
            optimizer.step()
        model.eval()
        for yield_valid_batch, input_valid_batch in valid_iter:
            yield_valid_batch_hat, instance_identity_batch = model(input_valid_batch)
            loss_batch = criterion(yield_valid_batch_hat, yield_valid_batch)
            metric.add(loss_batch.detach(), 1)
        valid_metric = metric[0] / metric[1]
        metric.reset()
        if best_loss is None or best_loss > valid_metric:
            best_loss = valid_metric
            early_stopping_counter = 0
            model_dict = model.state_dict()
        else:
            early_stopping_counter +=1
            if early_stopping_counter >= patience:
                best_loss = None
                break
    model.load_state_dict(model_dict)
    train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
    test_r2, test_mse, test_mae, yield_test_mse, test_instance_scores = presenter(model, test_iter, yield_test)
    train_accuracy_gather.append([train_r2, train_mse, train_mae])
    test_accuracy_gather.append([test_r2, test_mse, test_mae])
    train_instance_gather.append([yield_train_mse, train_instance_scores])
    test_instance_gather.append([yield_test_mse, test_instance_scores])
    return train_accuracy_gather, test_accuracy_gather, train_instance_gather, test_instance_gather, feature_importance