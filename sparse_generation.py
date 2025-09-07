from pathlib import Path
import pandas as pd
import random
import numpy as np
import torch

# 设置随机种子
random.seed(2025)
np.random.seed(2025)

# 获取rtMatrix存放路径
rtMatrix_csv = "data/QoS_Dataset/rtMatrix.csv"
# 读取rtMatrix
rtMatrix_df = pd.read_csv(rtMatrix_csv, header=None)
# -1转为0
rtMatrix_df.replace(-1, 0, inplace=True)
# df转换为数组形式
rtMatrix_array = rtMatrix_df.values

# 获取tpMatrix存放路径
tpMatrix_csv = "data/QoS_Dataset/tpMatrix.csv"
# 读取 tpMatrix
tpMatrix_df = pd.read_csv(tpMatrix_csv, header=None)
# -1转为0
tpMatrix_df.replace(-1, 0, inplace=True)
# df转换为数组形式
tpMatrix_array = tpMatrix_df.values

# 不再需要两个矩阵都不为零的过滤条件
# 保留原始矩阵的数据，不做交集处理

# 获取稀疏矩阵存放目录
# 1%稀疏度
rtMatrix1_csv = "data/QoS_Dataset/sparse1/rtMatrix1.csv"
# 99%稀疏度
rtMatrix99_csv = "data/QoS_Dataset/sparse1/rtMatrix99.csv"
# 2%稀疏度
rtMatrix2_csv = "data/QoS_Dataset/sparse1/rtMatrix2.csv"
# 98%稀疏度
rtMatrix98_csv = "data/QoS_Dataset/sparse1/rtMatrix98.csv"
# 3%稀疏度
rtMatrix3_csv = "data/QoS_Dataset/sparse1/rtMatrix3.csv"
# 97%稀疏度
rtMatrix97_csv = "data/QoS_Dataset/sparse1/rtMatrix97.csv"
# 4%稀疏度
rtMatrix4_csv = "data/QoS_Dataset/sparse1/rtMatrix4.csv"
# 96%稀疏度
rtMatrix96_csv = "data/QoS_Dataset/sparse1/rtMatrix96.csv"
# 5%稀疏度
rtMatrix5_csv = "data/QoS_Dataset/sparse/rtMatrix5.csv"
# 95%稀疏度
rtMatrix95_csv = "data/QoS_Dataset/sparse/rtMatrix95.csv"
# 10%稀疏度
rtMatrix10_csv = "data/QoS_Dataset/sparse/rtMatrix10.csv"
# 90%稀疏度
rtMatrix90_csv = "data/QoS_Dataset/sparse/rtMatrix90.csv"
# 15%稀疏度
rtMatrix15_csv = "data/QoS_Dataset/sparse/rtMatrix15.csv"
# 85%稀疏度
rtMatrix85_csv = "data/QoS_Dataset/sparse/rtMatrix85.csv"
# 20%稀疏度
rtMatrix20_csv = "data/QoS_Dataset/sparse/rtMatrix20.csv"
# 80%稀疏度
rtMatrix80_csv = "data/QoS_Dataset/sparse/rtMatrix80.csv"

# 1%稀疏度
tpMatrix1_csv = "data/QoS_Dataset/sparse1/tpMatrix1.csv"
# 99%稀疏度
tpMatrix99_csv = "data/QoS_Dataset/sparse1/tpMatrix99.csv"
# 2%稀疏度
tpMatrix2_csv = "data/QoS_Dataset/sparse1/tpMatrix2.csv"
# 98%稀疏度
tpMatrix98_csv = "data/QoS_Dataset/sparse1/tpMatrix98.csv"
# 3%稀疏度
tpMatrix3_csv = "data/QoS_Dataset/sparse1/tpMatrix3.csv"
# 97%稀疏度
tpMatrix97_csv = "data/QoS_Dataset/sparse1/tpMatrix97.csv"
# 4%稀疏度
tpMatrix4_csv = "data/QoS_Dataset/sparse1/tpMatrix4.csv"
# 96%稀疏度
tpMatrix96_csv = "data/QoS_Dataset/sparse1/tpMatrix96.csv"
# 5%稀疏度
tpMatrix5_csv = "data/QoS_Dataset/sparse/tpMatrix5.csv"
# 95%稀疏度
tpMatrix95_csv = "data/QoS_Dataset/sparse/tpMatrix95.csv"
# 10%稀疏度
tpMatrix10_csv = "data/QoS_Dataset/sparse/tpMatrix10.csv"
# 90%稀疏度
tpMatrix90_csv = "data/QoS_Dataset/sparse/tpMatrix90.csv"
# 15%稀疏度
tpMatrix15_csv = "data/QoS_Dataset/sparse/tpMatrix15.csv"
# 85%稀疏度
tpMatrix85_csv = "data/QoS_Dataset/sparse/tpMatrix85.csv"
# 20%稀疏度
tpMatrix20_csv = "data/QoS_Dataset/sparse/tpMatrix20.csv"
# 80%稀疏度
tpMatrix80_csv = "data/QoS_Dataset/sparse/tpMatrix80.csv"


# 稀疏矩阵生成函数 - 修改为单独处理RT和TP矩阵
def Generate_Sparse_Matrix(density, overlap_ratio=0.3):
    """
    生成稀疏矩阵，并控制rt和tp矩阵交集的比例

    参数:
    density: 稀疏度百分比，表示要保留的非零元素比例
    overlap_ratio: rt和tp矩阵交集元素占比，范围[0,1]，0表示无交集，1表示完全交集
    """
    # 获取原始矩阵的形状（行数和列数）
    rows, cols = rtMatrix_array.shape

    # 创建一个和原始矩阵同样形状的矩阵，用来存储分割后的训练、测试数据
    rtMatrix_train_array = np.zeros_like(rtMatrix_array)
    rtMatrix_test_array = np.zeros_like(rtMatrix_array)
    tpMatrix_train_array = np.zeros_like(tpMatrix_array)
    tpMatrix_test_array = np.zeros_like(tpMatrix_array)

    # 获取RT矩阵中非零元素的索引
    rt_nonzero_indices = [(i, j) for i in range(rows) for j in range(cols) if rtMatrix_array[i, j] != 0]
    # 获取TP矩阵中非零元素的索引
    tp_nonzero_indices = [(i, j) for i in range(rows) for j in range(cols) if tpMatrix_array[i, j] != 0]

    # 打乱索引顺序
    random.shuffle(rt_nonzero_indices)
    random.shuffle(tp_nonzero_indices)

    # 计算RT和TP矩阵训练集中应包含的元素数量
    rt_train_size = int(density / 100 * len(rt_nonzero_indices))
    tp_train_size = int(density / 100 * len(tp_nonzero_indices))

    # 计算应该有多少交集元素
    # 首先找出所有可能的交集（即在rt和tp中都有非零值的位置）
    all_possible_overlaps = list(set(rt_nonzero_indices).intersection(set(tp_nonzero_indices)))

    # 计算训练集中应该有的交集元素数量
    target_overlap_size = int(min(rt_train_size, tp_train_size) * overlap_ratio)
    actual_overlap_size = min(target_overlap_size, len(all_possible_overlaps))

    # 打乱可能的交集索引
    random.shuffle(all_possible_overlaps)

    # 选择交集元素
    overlap_indices = all_possible_overlaps[:actual_overlap_size]

    # 剩余需要填充的rt和tp元素数量
    remaining_rt = rt_train_size - actual_overlap_size
    remaining_tp = tp_train_size - actual_overlap_size

    # 从剩余非零元素中选择元素填充
    rt_only_indices = list(set(rt_nonzero_indices) - set(overlap_indices))
    tp_only_indices = list(set(tp_nonzero_indices) - set(overlap_indices))

    random.shuffle(rt_only_indices)
    random.shuffle(tp_only_indices)

    rt_selected = rt_only_indices[:remaining_rt]
    tp_selected = tp_only_indices[:remaining_tp]

    # 现在我们有了三组索引：overlap_indices, rt_selected, tp_selected

    # 填充rt训练矩阵（包括交集部分和rt特有部分）
    for i, j in overlap_indices + rt_selected:
        rtMatrix_train_array[i, j] = rtMatrix_array[i, j]

    # 填充tp训练矩阵（包括交集部分和tp特有部分）
    for i, j in overlap_indices + tp_selected:
        tpMatrix_train_array[i, j] = tpMatrix_array[i, j]

    # 生成测试矩阵 - 所有非零元素减去训练集元素
    rt_train_indices = set(overlap_indices + rt_selected)
    tp_train_indices = set(overlap_indices + tp_selected)

    for i, j in set(rt_nonzero_indices) - rt_train_indices:
        rtMatrix_test_array[i, j] = rtMatrix_array[i, j]

    for i, j in set(tp_nonzero_indices) - tp_train_indices:
        tpMatrix_test_array[i, j] = tpMatrix_array[i, j]

    # 数组转DataFrame
    rtMatrix_train_df = pd.DataFrame(rtMatrix_train_array)
    rtMatrix_test_df = pd.DataFrame(rtMatrix_test_array)
    tpMatrix_train_df = pd.DataFrame(tpMatrix_train_array)
    tpMatrix_test_df = pd.DataFrame(tpMatrix_test_array)

    # 计算测试集中rt和tp的交集
    rt_test_indices = set(rt_nonzero_indices) - rt_train_indices
    tp_test_indices = set(tp_nonzero_indices) - tp_train_indices
    test_overlap_indices = rt_test_indices.intersection(tp_test_indices)
    test_overlap_count = len(test_overlap_indices)

    # 将DataFrame保存为CSV文件
    if density == 1:
        rtMatrix_train_df.to_csv(rtMatrix1_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix99_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix1_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix99_csv, index=False, header=False)

    elif density == 2:
        rtMatrix_train_df.to_csv(rtMatrix2_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix98_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix2_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix98_csv, index=False, header=False)

    elif density == 3:
        rtMatrix_train_df.to_csv(rtMatrix3_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix97_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix3_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix97_csv, index=False, header=False)

    elif density == 4:
        rtMatrix_train_df.to_csv(rtMatrix4_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix96_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix4_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix96_csv, index=False, header=False)

    elif density == 5:
        rtMatrix_train_df.to_csv(rtMatrix5_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix95_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix5_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix95_csv, index=False, header=False)

    elif density == 10:
        rtMatrix_train_df.to_csv(rtMatrix10_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix90_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix10_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix90_csv, index=False, header=False)

    elif density == 15:
        rtMatrix_train_df.to_csv(rtMatrix15_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix85_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix15_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix85_csv, index=False, header=False)

    elif density == 20:
        rtMatrix_train_df.to_csv(rtMatrix20_csv, index=False, header=False)
        rtMatrix_test_df.to_csv(rtMatrix80_csv, index=False, header=False)
        tpMatrix_train_df.to_csv(tpMatrix20_csv, index=False, header=False)
        tpMatrix_test_df.to_csv(tpMatrix80_csv, index=False, header=False)


    # 统计非零元素和计算交集
    rt_train_count = np.count_nonzero(rtMatrix_train_array)
    rt_test_count = np.count_nonzero(rtMatrix_test_array)
    tp_train_count = np.count_nonzero(tpMatrix_train_array)
    tp_test_count = np.count_nonzero(tpMatrix_test_array)

    train_overlap_count = len(overlap_indices)

    # 打印统计信息
    print(f"密度: {density}%")
    print(f"RT训练集元素数: {rt_train_count}")
    print(f"RT测试集元素数: {rt_test_count}")
    print(f"TP训练集元素数: {tp_train_count}")
    print(f"TP测试集元素数: {tp_test_count}")
    print(f"训练集交集元素数: {train_overlap_count}")
    print(f"测试集交集元素数: {test_overlap_count}")
    print(f"训练集交集比例: {train_overlap_count / min(rt_train_count, tp_train_count):.4f}")
    print(f"测试集交集比例: {test_overlap_count / min(rt_test_count, tp_test_count) if min(rt_test_count, tp_test_count) > 0 else 0:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    # 输出原始RT和TP矩阵的非零元素总数
    rt_total = np.count_nonzero(rtMatrix_array)
    tp_total = np.count_nonzero(tpMatrix_array)
    total_overlap = len(set([(i, j) for i in range(rtMatrix_array.shape[0]) for j in range(rtMatrix_array.shape[1])
                             if rtMatrix_array[i, j] != 0]).intersection(
        set([(i, j) for i in range(tpMatrix_array.shape[0]) for j in range(tpMatrix_array.shape[1])
             if tpMatrix_array[i, j] != 0])))

    print(f"原始RT矩阵非零元素总数: {rt_total}")
    print(f"原始TP矩阵非零元素总数: {tp_total}")
    print(f"原始RT和TP矩阵交集元素总数: {total_overlap}")
    print(f"原始交集占比: {total_overlap / min(rt_total, tp_total):.4f}")
    print("=" * 50)

    Generate_Sparse_Matrix(1, overlap_ratio=0.5)
    Generate_Sparse_Matrix(2, overlap_ratio=0.5)
    Generate_Sparse_Matrix(3, overlap_ratio=0.5)
    Generate_Sparse_Matrix(4, overlap_ratio=0.5)
    # Generate_Sparse_Matrix(5, overlap_ratio=0.3)
    # Generate_Sparse_Matrix(10, overlap_ratio=0.3)
    # Generate_Sparse_Matrix(15, overlap_ratio=0.3)
    # Generate_Sparse_Matrix(20, overlap_ratio=0.3)