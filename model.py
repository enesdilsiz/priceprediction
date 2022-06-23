# Gerekli Kütüphaneleri Çağırma
import itertools
import warnings
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import plotly.express as px

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Pandas, gerekli ayarlamalar

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Veri setini okuma

train = pd.read_csv("DATATHON 2022/datasets/train.csv", parse_dates=['tarih'])
test = pd.read_csv("DATATHON 2022/datasets/testFeatures.csv", parse_dates=['tarih'])

# Harici veri setlerini okuma

doviz = pd.read_csv("DATATHON 2022/datasets/harici/doviz.csv", parse_dates=['Yayinlandigi Tarih'])
enflasyon = pd.read_csv("DATATHON 2022/datasets/harici/enflasyon.csv", parse_dates=['Column1'])


doviz = doviz.resample('MS', on='Yayinlandigi Tarih').mean()
doviz['tarih'] = doviz.index

#enflasyon = enflasyon[['Column1','TÜFE (Aylık % Değişim)']]
enflasyon.columns = ['tarih','enflasyon_yillik','enflasyon_aylik']
enflasyon.sort_values('tarih', axis=0, inplace=True)
enflasyon['enflasyon_cum'] = enflasyon['enflasyon_aylik'].cumsum(axis = 0)

##################################################
# Statistical Methods
##################################################

products = train['ürün'].unique()

trainS = train.set_index('tarih')
trainS = trainS.drop(['ürün besin değeri','ürün kategorisi','ürün üretim yeri','market','şehir'], axis=1)

testS = test.set_index('tarih')
testS = testS.drop(['ürün besin değeri','ürün kategorisi','ürün üretim yeri','market','şehir'], axis=1)


##################################################
# DES Double Exponential Smoothing
##################################################


def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

des_pred = pd.DataFrame({'ürün fiyatı':[]})

predictions_des = {}

for i in products:

    temp = trainS[trainS["ürün"]==i]
    whole = temp.groupby(temp.index).mean()
    train = whole[:48]
    test = whole[48:]

    best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas, step=12)

    des_model = ExponentialSmoothing(whole, trend="add").fit(smoothing_level=best_alpha,
                                                             smoothing_slope=best_beta)

    y_pred = des_model.forecast(12).to_frame().rename(columns={0: 'ürün fiyatı'})
    y_pred['ürün'] = i
    predictions_des[i] = y_pred
    des_pred = des_pred.append(y_pred)

# Plot DES
def plot_prediction_des(predictions, product):
    temp = trainS[trainS["ürün"] == product]
    whole = temp.groupby(temp.index).mean()
    whole.plot(legend=True, label="TRAIN", color=['blue'])
    predictions[product]['ürün fiyatı'].plot(legend=True, label="tahmini fiyat", color=['red'])
    plt.title(f"DES ile {product} Fiyat Tahmini")
    plt.show(block=True)

plot_prediction_des(predictions_des, 'kıyma')

# Export DES results to CSV
sample1 = testS.reset_index()
sample2 = des_pred.reset_index()
sample2 = sample2.rename(columns={'index': 'tarih'})

sample = sample1.merge(sample2, on=['ürün','tarih'], how='left')
des_submission = sample[['id','ürün fiyatı']]

#des_submission.to_csv("submissionv05.csv", index=False)


##################################################
# TES Triple Exponential Smoothing (Holt-Winters)
##################################################

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

tes_pred = pd.DataFrame({'ürün fiyatı':[]})
predictions_tes = {}

for i in products:

    temp = trainS[trainS["ürün"]==i]
    whole = temp.groupby(temp.index).mean()
    train = whole[:48]
    test = whole[48:]

    best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg, step=12)

    tes_model = ExponentialSmoothing(whole, trend="mul", seasonal="mul", seasonal_periods=12). \
        fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

    y_pred = tes_model.forecast(12).to_frame().rename(columns={0: 'ürün fiyatı'})
    y_pred['ürün'] = i
    predictions_tes[i] = y_pred
    tes_pred = tes_pred.append(y_pred)

def plot_prediction_tes(predictions, product):
    temp = trainS[trainS["ürün"] == product]
    whole = temp.groupby(temp.index).mean()
    whole.plot(legend=True, label="TRAIN", color=['blue'])
    predictions[product]['ürün fiyatı'].plot(legend=True, label="tahmini fiyat", color=['red'])
    plt.title(f"TES ile {product} Fiyat Tahmini")
    plt.show(block=True)

plot_prediction_tes(predictions_tes, 'kıyma')

sample1 = testS.reset_index()
sample2 = tes_pred.reset_index()
sample2 = sample2.rename(columns={'index': 'tarih'})

sample=sample1.merge(sample2, on=['ürün','tarih'], how='left')
tes_submission = sample[['id','ürün fiyatı']]
tes_final = sample[['tarih','ürün','ürün fiyatı']].set_index('tarih')

#tes_submission.to_csv("submissionv08.csv", index=False)

results = sample2.set_index('tarih')

results.to_csv('results.csv')
trainS.to_csv('trainS.csv')



##################################################
# ARIMA
##################################################

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params


interval = pd.date_range('2021-01-01', periods=12, freq='MS')
arima_pred = pd.DataFrame({'ürün fiyatı':[]})
predictions_arima = {}

for i in products:

    temp = trainS[trainS["ürün"]==i]
    whole = temp.groupby(temp.index).mean()
    train = whole[:48]
    test = whole[48:]

    best_params_aic = arima_optimizer_aic(train, pdq)

    arima_model = ARIMA(whole, best_params_aic).fit(disp=0)

    y_pred = arima_model.forecast(12)[0]
    y_pred_df = pd.DataFrame(y_pred, columns=['ürün fiyatı'], index = interval)
    y_pred_df['ürün'] = i
    predictions_arima[i] = y_pred
    arima_pred = arima_pred.append(y_pred_df)

sample1 = testS.reset_index()
sample2 = arima_pred.reset_index()
sample2 = sample2.rename(columns={'index': 'tarih'})

sample = sample1.merge(sample2, on=['ürün', 'tarih'], how='left')
arima_submission = sample[['id', 'ürün fiyatı']]

#arima_submission.to_csv("submissionv06.csv", index=False)

##################################################
# SARIMA
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order


interval = pd.date_range('2021-01-01', periods=12, freq='MS')
sarima_pred = pd.DataFrame({'ürün fiyatı':[]})
predictions_sarima = {}

for i in products:

    temp = trainS[trainS["ürün"]==i]
    whole = temp.groupby(temp.index).mean()
    train = whole[:48]
    test = whole[48:]

    best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

    model = SARIMAX(whole, order=best_order, seasonal_order=best_seasonal_order)
    sarima_final_model = model.fit(disp=0)
    y_pred_test = sarima_final_model.get_forecast(steps=12)
    y_pred = y_pred_test.predicted_mean

    y_pred_df = pd.DataFrame(y_pred, columns=['ürün fiyatı'], index = interval)
    y_pred_df['ürün'] = i
    predictions_sarima[i] = y_pred
    sarima_pred = sarima_pred.append(y_pred_df)

sample1 = testS.reset_index()
sample2 = sarima_pred.reset_index()
sample2 = sample2.rename(columns={'index': 'tarih'})

sample = sample1.merge(sample2, on=['ürün', 'tarih'], how='left')
sarima_submission = sample[['id', 'ürün fiyatı']]

#sarima_submission.to_csv("submissionv07.csv", index=False)




##################################

train = pd.read_csv("DATATHON 2022/datasets/train.csv", parse_dates=['tarih'])
test = pd.read_csv("DATATHON 2022/datasets/testFeatures.csv", parse_dates=['tarih'])
df = pd.concat([train, test], axis=0)

df.shape
df.head()
df.info()
df.isnull().sum()
df.describe().T

len(train['ürün'].unique())
len(train['ürün kategorisi'].unique())

doviz.shape
doviz.info()
doviz.head()


df_new = pd.merge(df, doviz, on='tarih')
df = pd.merge(df_new, enflasyon, on='tarih')


df.isnull().sum()
df.describe().T
df.head()


df['ürün'].unique()
df["ürün"].value_counts()
df["şehir"].unique()
df["ürün kategorisi"].value_counts()


df["tarih"].min(), df["tarih"].max()

# Korelasyon Matrisi
df_corr = df[['ürün besin değeri','ürün fiyatı','Doviz Satis','enflasyon_yillik','enflasyon_aylik','enflasyon_cum']]
df_corr.columns = ['besin değeri','ürün fiyatı','dolar kuru','yıllık enflasyon','aylık enflasyon','kümülatif enflasyon']
fig = px.imshow(df_corr.corr(), color_continuous_scale='RdBu')
fig.write_image("heatmap.png", width=1024, height=1024)

##################################################
# FEATURE ENGINEERING
##################################################

#df['pandemi'] = False
#df.loc[df['tarih'] >= '2020-04-01', 'pandemi'] = True


def create_date_features(df):
    df['month'] = df['tarih'].dt.month
    df['year'] = df['tarih'].dt.year
    df['day_of_year'] = df['tarih'].dt.dayofyear

    return df

df = create_date_features(df)


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['ürün_fiyat_lag_' + str(lag)] = dataframe.groupby(["ürün", "ürün kategorisi", "ürün üretim yeri", "şehir", "market"])['ürün fiyatı'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [1, 2, 3, 4, 5, 6])

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['fiyat_roll_mean_' + str(window)] = dataframe.groupby(["ürün", "ürün kategorisi", "ürün üretim yeri", "şehir", "market"])['ürün fiyatı']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [1, 2, 3, 4, 5, 6])

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['fiyat_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["ürün", "ürün kategorisi", "ürün üretim yeri", "şehir", "market"])['ürün fiyatı'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [1, 2, 3, 4, 5]

df = ewm_features(df, alphas, lags)


##################################################
# LABEL ENCODING
##################################################

df = pd.get_dummies(df, columns=['ürün', 'ürün kategorisi', 'ürün üretim yeri', 'market', 'şehir', 'month', 'year', 'day_of_year'], drop_first=True)

##################################################
# LGBM - Light Gradient Boosting Machines
##################################################

df['ürün fiyatı'] = np.log1p(df["ürün fiyatı"].values)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False



train = df.loc[df['tarih'] <= '2020-11-01']
valid = df.loc[(df['tarih'] > '2020-11-01') & (df['tarih'] <= '2020-12-01')]
test = df.loc[df['tarih'] > '2020-12-01']

cols = [col for col in train.columns if col not in ['tarih', 'ürün fiyatı','id']]


Y_train = train['ürün fiyatı']
X_train = train[cols]

Y_valid = valid['ürün fiyatı']
X_valid = valid[cols]

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 10000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_valid, label=Y_valid, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)


Y_train.shape, X_train.shape, Y_valid.shape, X_valid.shape

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=30, plot=True)


X_test = test[cols]

test_preds = model.predict(X_test, num_iteration=model.best_iteration)

submission_df = test.loc[:, ["id", "ürün fiyatı"]]
submission_df['ürün fiyatı'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submissionv02.csv", index=False)


##################################################
# LGBM Regressor
##################################################

lgbreg = lgb.LGBMRegressor(n_estimators=1000, max_depth=5, random_state=0, learning_rate=0.01)
lgbreg.fit(X_train,Y_train)
lgbreg.score(X_train,Y_train)
y_pred = model.predict(X_valid)
sm = smape(np.expm1(y_pred), np.expm1(Y_valid))

from sklearn.metrics import mean_squared_error

rmse_score = np.sqrt(mean_squared_error(Y_valid, y_pred))

test_pred = model.predict(X_test)

submission_df = test.loc[:, ["id", "ürün fiyatı"]]
submission_df['ürün fiyatı'] = np.expm1(test_pred)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submissionv03.csv", index=False)