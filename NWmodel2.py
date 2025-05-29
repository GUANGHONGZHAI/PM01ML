import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import joblib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def load_nc_data(file_path):
    """加载NetCDF文件"""
    try:
        with nc.Dataset(file_path, 'r') as nc_data:
            PM01_DRY = nc_data.variables['PM01_DRY'][:]
            T_MAX = nc_data.variables['T_MAX'][:]
            T_MIN = nc_data.variables['T_MIN'][:]
            T2 = nc_data.variables['T2'][:]
            LU_INDEX = nc_data.variables['LU_INDEX'][:]
            U = nc_data.variables['U'][:]
            V = nc_data.variables['V'][:]
            PBLH = nc_data.variables['PBLH'][:]
            RH = nc_data.variables['RH'][:]
            PR = nc_data.variables['PR'][:]
            PSFC = nc_data.variables['PSFC'][:]
            
            # 处理U和V的形状，使其与其他变量一致
            target_shape = (1, 99, 99)
            
            if U.shape != target_shape:
                U = U[:, :, :99]
            
            if V.shape != target_shape:
                V = V[:, :99, :]

            # 读取全局变量中的星期数变量
            if 'WEEK' in nc_data.ncattrs():
                WEEK = nc_data.getncattr('WEEK')
                WEEK = np.full((99, 99), WEEK)
            else:
                print(f"文件 {file_path} 中未找到 'WEEK' 全局变量")
                return None

            return {
                'PM01_DRY': PM01_DRY,
                'T_MAX': T_MAX,
                'T_MIN': T_MIN,
                'T2': T2,
                'LU_INDEX': LU_INDEX,
                'U': U,
                'V': V,
                'PBLH': PBLH,
                'RH': RH,
                'PR': PR,
                'PSFC': PSFC,
                'WEEK': WEEK,
            }
    except Exception as e:
        print(f"加载文件 {file_path} 时出错：{e}")
        return None

# 指定数据目录
base_dir = '/work/home/zhaiguanghong/apprepo/DATA/Result18301/avg'

# 查找所有 .nc 文件
nc_files = glob.glob(os.path.join(base_dir, '*.nc'))

print(f"找到 {len(nc_files)} 个 NetCDF 文件")

data = []
for file_path in nc_files:
    file_data = load_nc_data(file_path)
    if file_data is not None:
        # 创建一个字典用于存储展平后的数据
        flattened_data = {key: value.flatten() for key, value in file_data.items()}
        
        df = pd.DataFrame({
            'PM01_DRY': file_data['PM01_DRY'].flatten(),
            'T_MAX': file_data['T_MAX'].flatten(),
            'T_MIN': file_data['T_MIN'].flatten(),
            'T2': file_data['T2'].flatten(),
            'LU_INDEX': file_data['LU_INDEX'].flatten(),
            'U': file_data['U'].flatten(),
            'V': file_data['V'].flatten(),
            'PBLH': file_data['PBLH'].flatten(),
            'RH': file_data['RH'].flatten(),
            'PR': file_data['PR'].flatten(),
            'PSFC': file_data['PSFC'].flatten(),
            'WEEK': file_data['WEEK'].flatten(),
        })
        
        data.append(df)

# 合并所有数据
if data:
    data = pd.concat(data, ignore_index=True)
else:
    print("没有成功加载任何数据文件")
    exit()

# 处理缺失值
data.dropna(inplace=True)

# 定义自变量和因变量
X = data[['T_MAX', 'T_MIN', 'T2', 'LU_INDEX', 'U', 'V', 'PBLH', 'RH', 'PR', 'PSFC', 'WEEK']]
y = data['PM01_DRY']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用递归特征消除（RFE）选择最佳自变量子集
selector = RFE(estimator=LinearRegression(), n_features_to_select=5, step=1)
selector.fit(X_scaled, y)

# 选择最佳自变量子集
X_selected = selector.transform(X_scaled)

# 定义MLPRegressor的参数范围，用于GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(16,), (32,), (64,), (16, 16), (32, 32), (64, 64)],
    'activation': ['relu'],
    'solver': ['adam'],
    'learning_rate': ['adaptive'],
    'max_iter': [500],  # 增加最大迭代次数
    'alpha': [0.001, 0.01, 0.1],  # 增加正则化参数范围
    'learning_rate_init': [0.001, 0.01, 0.1],  # 手动设置初始学习率
}

# 使用十折交叉验证确定最优参数
mlp = MLPRegressor(random_state=42)
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
grid_search.fit(X_selected, y)

# 输出最优参数和模型
best_mlp = grid_search.best_estimator_
print("最优参数：", grid_search.best_params_)
print("最优模型：", best_mlp)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 使用最优模型进行训练
best_mlp.fit(X_train, y_train)

# 保存训练好的模型和标准化器
model_save_path = 'mlp_model.pkl'
joblib.dump(best_mlp, model_save_path)
print(f"模型已保存为：{model_save_path}")

scaler_save_path = 'scaler.pkl'
joblib.dump(scaler, scaler_save_path)
print(f"标准化器已保存为：{scaler_save_path}")