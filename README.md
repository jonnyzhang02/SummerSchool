<!--
 * @Author       : JonnyZhang 71881972+jonnyzhang02@users.noreply.github.com
 * @LastEditTime : 2023-09-12 10:02
 * @FilePath     : \SummerSchool\README.md
 * 
 * coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
-->
# Task 1

## 一、目标检测的一般流程

目标检测是计算机视觉领域的关键任务，旨在从图像或视频中准确地定位和识别出不同类别的目标物体。其一般流程包括以下步骤：

1. 数据收集和预处理：收集包含目标物体的图像或视频数据，并进行预处理操作，如图像缩放、裁剪和色彩空间转换，以确保数据的一致性和适应性。

2. 特征提取：使用计算机视觉技术从输入图像中提取有意义的特征。常用的特征提取方法包括基于手工设计的特征（如Haar特征、HOG特征）和基于深度学习的特征（如卷积神经网络中的卷积层输出）。

3. 候选区域生成：在图像中生成一组**候选目标区域**，这些区域可能包含目标物体。候选区域生成方法可以是基于滑动窗口的方法，也可以是基于区域提议的方法（如Selective Search、EdgeBoxes等）。

4. 目标分类：对每个候选区域进行目标类别的分类。这一步骤可以使用机器学习算法（如支持向量机、随机森林等）或深度学习算法（如卷积神经网络）进行目标分类。

5. 边界框回归：对于分类为目标的候选区域，进一步精细调整其位置和大小，以更准确地框定目标的边界框。这一步骤通常使用回归模型来预测边界框的坐标偏移量。

6. 后处理：根据一些规则和策略对检测结果进行后处理，包括去除重叠的边界框、筛选置信度较低的检测结果等。

7. 输出结果：输出目标检测的结果，通常以边界框的形式给出，包括目标的类别标签、位置和置信度等信息。

## 二、YOLOv1论文阅读

