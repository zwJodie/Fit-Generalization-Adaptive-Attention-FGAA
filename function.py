### Import required libraries
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import IntegratedGradients
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.nonparametric.smoothers_lowess import lowess
import copy
import time

### Select GPU when available; otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### Construct the eight environmental features used by the models
def natural_feature_builder(natural_dataset):
    feature_list0 = ["temperature_2m", "dewpoint_temperature_2m", "surface_solar_radiation_downwards_sum", "surface_pressure", "total_precipitation_sum"]
    natural_featured = natural_dataset[feature_list0].copy()
    soil_temperature = natural_dataset[["soil_temperature_level_1", "soil_temperature_level_2", "soil_temperature_level_3"]].mean(1)
    volumetric_soil_water = natural_dataset[["volumetric_soil_water_layer_1", "volumetric_soil_water_layer_2", "volumetric_soil_water_layer_3"]].mean(1)
    wind_speed_10m = (natural_dataset["u_component_of_wind_10m"] ** 2 + natural_dataset["v_component_of_wind_10m"] ** 2) ** (1 / 2)
    natural_featured["soil_temperature"] = soil_temperature
    natural_featured["volumetric_soil_water"] = volumetric_soil_water
    natural_featured["wind_speed_10m"] = wind_speed_10m
    feature_list1 = ["surface_solar_radiation_downwards_sum", "temperature_2m", "soil_temperature", "dewpoint_temperature_2m"]
    feature_list2 = ["volumetric_soil_water", "total_precipitation_sum", "surface_pressure", "wind_speed_10m"]
    natural_feature = natural_featured[feature_list1 + feature_list2]
    return natural_feature


### Apply a common completeness mask and retain counties with at least 10 complete years
def filter_(yield_data, social_data, natural_data, county_index):
    yield_flat = yield_data.reshape(-1)
    social_flat, natural_flat, index_nan = social_data.reshape(yield_flat.size(0), -1), natural_data.reshape(yield_flat.size(0), -1), []
    for i in range(yield_flat.size(0)):
        if torch.isnan(yield_flat[i]) or torch.isnan(social_flat[i, :]).any() or torch.isnan(natural_flat[i, :]).any():
            index_nan.append(i)
    yield_flat[index_nan] = np.nan
    yield_data = yield_flat.reshape(-1, 22)
    social_data, natural_data = social_flat.reshape(yield_data.size(0), 22, -1), natural_flat.reshape(yield_data.size(0), 22, -1)
    yield_data_csv = pd.DataFrame(yield_data.numpy())
    yield_notnan_bool = ~yield_data_csv.T.isna()
    county_notnan = np.arange(yield_data.size(0))[yield_notnan_bool.sum() >= 10]
    yield_notnan, social_notnan, natural_notnan = yield_data[county_notnan], social_data[county_notnan], natural_data[county_notnan]
    if county_index is not None:
        county_notnan_index = county_index[county_notnan]
        return yield_notnan, social_notnan, natural_notnan, county_notnan_index
    else:
        return yield_notnan, social_notnan, natural_notnan


### Convert nine raw socioeconomic indicators into eight features
def social_feature_builder(social_flat):
    social_feature = torch.zeros([social_flat.shape[0], 8])
    social_feature[:, 0] = social_flat[:, 1] / social_flat[:, 0]  # Population Density
    social_feature[:, 1] = social_flat[:, 8] / social_flat[:, 1]  # Health-care Resource
    social_feature[:, 2] = social_flat[:, 7] / social_flat[:, 1]  # Pupil Proportion
    social_feature[:, 3] = social_flat[:, 6] / social_flat[:, 1]  # Teenager Proportion
    social_feature[:, 4] = social_flat[:, 2] / social_flat[:, 1]  # Fiscal Revenue
    social_feature[:, 5] = social_flat[:, 3] / social_flat[:, 1]  # Fiscal Expenditure
    social_feature[:, 6] = social_flat[:, 4] / social_flat[:, 1]  # Residential Savings
    social_feature[:, 7] = social_flat[:, 5] / social_flat[:, 1]  # Institutional Loans
    return social_feature


### Flatten county-year arrays and retain observations with non-missing yield
def flatten(yield_data, social_data, natural_data, county_index):
    yield_flat, social_flat, natural_flat = yield_data.reshape(-1), social_data.reshape(-1, 9), natural_data.reshape(-1, 72)
    index_notnan = np.arange(yield_flat.size(0))[~torch.isnan(yield_flat)]
    yield_flat, social_flat, natural_flat = yield_flat[index_notnan].reshape(-1, 1), social_feature_builder(social_flat[index_notnan, :]), natural_flat[index_notnan, :]
    if county_index is not None:
        county_year_index = np.concatenate([np.repeat(county_index, 22, 0), np.tile(np.arange(22), len(county_index)).reshape(-1, 1)], 1)[index_notnan, :]
        return yield_flat, social_flat, natural_flat, county_year_index
    else:
        return yield_flat, social_flat, natural_flat


### Decompose yield observations into LOWESS trend and discrepancy yield
def resolve(yield_data):
    def lowess_(data):
        trend, years = np.full_like(data, np.nan, dtype=np.float64), np.arange(22)
        for i in range(data.shape[0]):
            data_county = data[i, :].numpy()
            mask_nan = np.isnan(data_county)
            valid_idx = np.where(~mask_nan)[0]
            start, end = valid_idx[0], valid_idx[-1]
            data_county, year = data_county[start : end + 1], years[start : end + 1]
            data_county = pd.Series(data_county).interpolate(limit_direction="both").to_numpy()
            data_trend, data_trend_valid = lowess(data_county, year, frac=0.5, it=3, return_sorted=False), np.full(end - start + 1, np.nan)
            data_trend_valid[~mask_nan[start : end + 1]] = data_trend[~mask_nan[start : end + 1]]
            trend[i, start : end + 1] = data_trend_valid
        trend = torch.from_numpy(trend).float()
        resid = data - trend
        data, trend, resid = data.reshape(-1), trend.reshape(-1), resid.reshape(-1)
        return data, trend, resid

    yield_obsvd, yield_trend, yield_resid = lowess_(yield_data)
    index_notnan = np.arange(yield_trend.size(0))[~torch.isnan(yield_trend)]
    yield_obsvd, yield_trend, yield_resid = yield_obsvd[index_notnan].reshape(-1, 1), yield_trend[index_notnan].reshape(-1, 1), yield_resid[index_notnan].reshape(-1, 1)
    return yield_obsvd, yield_trend, yield_resid


### Fit the standardization on training data and apply it to validation and test data
def scaler(data_train, data_valid, data_test):
    scaler = StandardScaler()
    scaler.fit(data_train)
    train_scaled, valid_scaled, test_scaled = scaler.transform(data_train), scaler.transform(data_valid), scaler.transform(data_test)
    train_scaled, valid_scaled, test_scaled = torch.from_numpy(train_scaled).float(), torch.from_numpy(valid_scaled).float(), torch.from_numpy(test_scaled).float()
    return train_scaled, valid_scaled, test_scaled


### Define the six supported deep learning architectures
class Model(nn.Module):
    def __init__(self, model_type, data_type, num_instance_identity, require_identity_output=False):
        super().__init__()
        input_size = {"baseline": 72, "addition": 80}.get(data_type)
        self.layer = {
            "rnn": nn.RNN(8, 72, num_layers=2, batch_first=True),
            "lst": nn.LSTM(8, 72, num_layers=2, batch_first=True),
            "gru": nn.GRU(8, 72, num_layers=2, batch_first=True),
            "cnn": nn.Conv1d(8, 8, kernel_size=3, padding=1),
            "att": nn.MultiheadAttention(8, 2, batch_first=True),
        }.get(model_type)
        self.network = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, num_instance_identity), nn.ReLU()
        )
        self.dense, self.model_type, self.data_type, self.require_identity_output = nn.Linear(num_instance_identity, 1), model_type, data_type, require_identity_output

    ### Perform architecture-specific feature extraction and yield prediction
    def forward(self, input_data, test_data=False):
        if self.model_type == "dnn":
            net_input = input_data
        else:
            net_input = input_data[:, 8:] if self.data_type == "addition" else input_data
            net_input = net_input.reshape(-1, 9, 8)
            if self.model_type in ["rnn", "lst", "gru"]:
                net_input = self.layer(net_input)[0].mean(dim=1)
            elif self.model_type == "cnn":
                net_input = self.layer(net_input.permute(0, 2, 1)).permute(0, 2, 1).reshape(-1, 72)
            elif self.model_type == "att":
                net_input = (net_input + self.layer(net_input, net_input, net_input, need_weights=False)[0]).reshape(-1, 72)
            if self.data_type == "addition":
                net_input = torch.cat([input_data[:, :8], net_input], dim=-1)
        mid_output = self.network(net_input)
        last_output = self.dense(mid_output)
        if self.require_identity_output:
            return last_output, mid_output
        return last_output


### Train the baseline model, calculate FGAA scores, and perform FGAA-weighted fine-tuning
def predictor(model_type, data_type, num_instance_identity, require_explanation, yield_train, social_train, natural_train,
yield_valid, social_valid, natural_valid, batch_size, learning_rate, max_epoch, patience, train_breaker, yield_test, social_test, natural_test):
    ### Calculate sample-level FGAA scores from scaled dot-product similarities
    def attention(instance_identity):
        distance = instance_identity.size(-1)
        instance_scores = torch.matmul(instance_identity, instance_identity.transpose(0, 1)) / torch.sqrt(torch.tensor(distance))
        instance_scores = instance_scores.mean(1)
        return instance_scores

    ### Compute aggregate metrics, sample-level errors, and FGAA scores
    def presenter(model, data_iter, yield_obsvd):
        instance_criterion, instance_identity, yield_hat, yield_mse = nn.MSELoss(reduction="none"), [], [], []
        for yield_batch, input_batch in data_iter:
            yield_batch_hat, instance_identity_batch = model(input_batch)
            yield_batch_mse = instance_criterion(yield_batch_hat, yield_batch)
            instance_identity.append(instance_identity_batch.detach())
            yield_hat.append(yield_batch_hat.detach().cpu())
            yield_mse.append(yield_batch_mse.detach().cpu())
        instance_identity = torch.concat(instance_identity)
        yield_hat, yield_mse = np.concatenate(yield_hat), np.concatenate(yield_mse)
        num_output = next(iter(data_iter))[0].shape[1]
        r2, mse, mae = r2_score(yield_obsvd.cpu(), yield_hat), mean_squared_error(yield_obsvd.cpu(), yield_hat), mean_absolute_error(yield_obsvd.cpu(), yield_hat)
        instance_scores = attention(instance_identity)
        return r2, mse, mae, yield_mse, instance_scores.cpu().numpy().reshape(-1, 1)

    ### Compute Integrated Gradients using an all-zero standardized baseline
    def explainer(model_type, data_type, num_instance_identity, model_dict, data_to_explain):
        model_to_explain = Model(model_type, data_type, num_instance_identity)
        model_to_explain.load_state_dict(model_dict)
        model_to_explain.eval()
        model_for_explain = IntegratedGradients(model_to_explain)
        feature_importance = []
        for yield_batch, input_batch in data_to_explain:
            feature_importance_batch = model_for_explain.attribute(input_batch.cpu(), baselines=torch.zeros(input_batch.size()))
            feature_importance.append(feature_importance_batch)
        feature_importance = torch.concat(feature_importance)
        return feature_importance.numpy()

    ### Initialize the selected model and assemble the requested feature set
    model = Model(model_type, data_type, num_instance_identity, True).to(device)
    if data_type == "baseline":
        input_train, input_valid, input_test = natural_train, natural_valid, natural_test
    elif data_type == "addition":
        input_train, input_valid = torch.cat([social_train, natural_train], -1), torch.cat([social_valid, natural_valid], -1)
        input_test = torch.cat([social_test, natural_test], -1)
    yield_train, yield_valid, yield_test = yield_train.to(device), yield_valid.to(device), yield_test.to(device)
    input_train, input_valid, input_test = input_train.to(device), input_valid.to(device), input_test.to(device)
    ### Keep sample order fixed so predictions, indices, and precomputed FGAA scores remain aligned
    train_iter = DataLoader(TensorDataset(yield_train, input_train), batch_size=batch_size, shuffle=False)
    valid_iter = DataLoader(TensorDataset(yield_valid, input_valid), batch_size=batch_size, shuffle=False)
    test_iter = DataLoader(TensorDataset(yield_test, input_test), batch_size=batch_size, shuffle=False)
    ### Configure optimization, early stopping, and result containers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion, instance_criterion, patience, best_loss, early_stopping_counter = nn.MSELoss(), nn.MSELoss(reduction="none"), patience, None, 0
    train_accuracy_gather, test_accuracy_gather, train_instance_gather, test_instance_gather, feature_importance = [], [], [], [], []
    test_r2, test_mse, test_mae, yield_test_hat, yield_test_mse, test_instance_scores = [], [], [], [], [], []

    ### Stage 1: Baseline model training.
    for epoch in range(max_epoch):
        model.train()
        for yield_train_batch, input_train_batch in train_iter:
            optimizer.zero_grad()
            yield_train_batch_hat, instance_identity_batch = model(input_train_batch)
            loss_batch = criterion(yield_train_batch_hat, yield_train_batch)
            loss_batch.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            metric_summer, batch_counter = 0.0, 0
            if epoch == 0 and train_breaker is True:
                ### Evaluate the once-trained model
                train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
                test_r2, test_mse, test_mae, yield_test_mse, test_instance_scores = presenter(model, test_iter, yield_test)
                train_accuracy_gather.append([train_r2, train_mse, train_mae])
                test_accuracy_gather.append([test_r2, test_mse, test_mae])
                train_instance_gather.append([yield_train_mse, train_instance_scores])
                test_instance_gather.append([yield_test_mse, test_instance_scores])
            for yield_valid_batch, input_valid_batch in valid_iter:
                yield_valid_batch_hat, instance_identity_batch = model(input_valid_batch)
                loss_batch = criterion(yield_valid_batch_hat, yield_valid_batch)
                metric_summer += loss_batch.item()
                batch_counter += 1
            valid_metric = metric_summer / batch_counter
            if best_loss is None or best_loss > valid_metric:
                best_loss, early_stopping_counter, model_dict = valid_metric, 0, copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    break
    ### Restore and evaluate the fully trained model
    model.load_state_dict(model_dict)
    train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
    test_r2, test_mse, test_mae, yield_test_mse, test_instance_scores = presenter(model, test_iter, yield_test)
    if require_explanation is True:
        feature_importance = explainer(model_type, data_type, num_instance_identity, model_dict, test_iter)
    train_accuracy_gather.append([train_r2, train_mse, train_mae])
    test_accuracy_gather.append([test_r2, test_mse, test_mae])
    train_instance_gather.append([yield_train_mse, train_instance_scores])
    test_instance_gather.append([yield_test_mse, test_instance_scores])

    ### Step 2: Sample-level FGAA score calculation.
    train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
    instance_scores, best_loss = torch.from_numpy(train_instance_scores.reshape(-1)).float().to(device), None
    train_iter_finetune = DataLoader(TensorDataset(yield_train, input_train, instance_scores), batch_size=batch_size, shuffle=False)

    ### Step 3:FGAA-weighted fine-tuning.
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
        with torch.no_grad():
            metric_summer, batch_counter = 0.0, 0
            for yield_valid_batch, input_valid_batch in valid_iter:
                yield_valid_batch_hat, instance_identity_batch = model(input_valid_batch)
                loss_batch = criterion(yield_valid_batch_hat, yield_valid_batch)
                metric_summer += loss_batch.item()
                batch_counter += 1
            valid_metric = metric_summer / batch_counter
            if best_loss is None or best_loss > valid_metric:
                best_loss, early_stopping_counter, model_dict = valid_metric, 0, copy.deepcopy(model.state_dict())
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    break
    ### Restore and evaluate the FGAA-enhanced model
    model.load_state_dict(model_dict)
    train_r2, train_mse, train_mae, yield_train_mse, train_instance_scores = presenter(model, train_iter, yield_train)
    test_r2, test_mse, test_mae, yield_test_mse, test_instance_scores = presenter(model, test_iter, yield_test)
    train_accuracy_gather.append([train_r2, train_mse, train_mae])
    test_accuracy_gather.append([test_r2, test_mse, test_mae])
    train_instance_gather.append([yield_train_mse, train_instance_scores])
    test_instance_gather.append([yield_test_mse, test_instance_scores])
    return train_accuracy_gather, test_accuracy_gather, train_instance_gather, test_instance_gather, feature_importance
