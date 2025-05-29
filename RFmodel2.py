import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import joblib

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
            # 假设其他变量的形状为(1, 99, 99)
            target_shape = (1, 99, 99)
            
            if U.shape != target_shape:
                # 裁剪U的第三个维度
                U = U[:, :, :99]
            
            if V.shape != target_shape:
                # 裁剪V的第二个维度
                V = V[:, :99, :]

            # 读取全局变量中的星期数变量
            if 'WEEK' in nc_data.ncattrs():
                WEEK = nc_data.getncattr('WEEK')
                # 将标量扩展为与空间维度一致的二维数组
                WEEK = np.full((99, 99), WEEK)
            else:
                print(f"文件 {file_path} 中未找到 'WEEK' 全局变量")
                return None

            
            # 打印变量形状
            print(f"文件：{file_path}")
            print(f"PM01_DRY形状：{PM01_DRY.shape}")
            print(f"T_MAX形状：{T_MAX.shape}")
            print(f"T_MIN形状：{T_MIN.shape}")
            print(f"T2形状：{T2.shape}")
            print(f"LU_INDEX形状：{LU_INDEX.shape}")
            print(f"U形状：{U.shape}")
            print(f"V形状：{V.shape}")
            print(f"PBLH形状：{PBLH.shape}")
            print(f"RH形状：{RH.shape}")
            print(f"PR形状：{PR.shape}")
            print(f"PSFC形状：{PSFC.shape}")
            print(f"WEEK形状：{PSFC.shape}")
            
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
        
        # 打印每个变量展平后的长度
        print(f"文件 {file_path} 的展平后变量长度：")
        for key, value in flattened_data.items():
            print(f"{key}: {len(value)}")
      
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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义随机森林模型
rf = RandomForestRegressor(random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],  # 决策树数量
    'max_features': ['sqrt', 'log2', None],  # 每个节点考虑的特征数
    'min_samples_split': [2, 5, 10],  # 节点分裂所需的最小样本数
    'max_depth': [None, 10, 20, 30]  # 树的最大深度
}

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳参数和模型
print("最佳参数组合：", grid_search.best_params_)
print("最佳模型得分：", grid_search.best_score_)

best_rf = grid_search.best_estimator_

# 保存最佳模型
model_save_path = 'best_rf_model.pkl'
joblib.dump(best_rf, model_save_path)
print(f"最佳模型已保存为：{model_save_path}")

# 特征重要性分析
importances = best_rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances)
plt.xlabel('Importance')
plt.ylabel('Variables')
plt.title('Feature Importance Analysis')
plt.savefig('Feature_Importance.png', bbox_inches='tight', dpi=300)
plt.close()