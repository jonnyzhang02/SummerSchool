<!--
 * @Author       : JonnyZhang 71881972+jonnyzhang02@users.noreply.github.com
 * @LastEditTime : 2023-09-11 14:06
 * @FilePath     : \SummerSchool\note.md
 * 
 * coded by ZhangYang@BUPT, my email is zhangynag0207@bupt.edu.cn
-->
# YOLO 系列

## 基础

![](assets/2023-09-11-10-02-08.png)

![](assets/2023-09-11-10-02-34.png)

![](assets/2023-09-11-10-03-14.png)

### 两阶段和一阶段

![](assets/2023-09-11-10-09-48.png)

![](assets/2023-09-11-08-46-57.png)

优势

Faster-RCNN 只有5FPS大概

![](assets/2023-09-11-08-47-52.png)

主要就是两个指标：精度map和速度FPS

![](assets/2023-09-11-08-51-52.png)


### IoU和Map指标

精度和recall一般是反比的，精度越高，recall越低

![](assets/2023-09-11-08-52-57.png)

IoU：交并比，交集/并集 Intersection over Union

![](assets/2023-09-11-08-54-42.png)

Precision：精度，预测为正的样本中，真实为正的样本的比例

Recall：召回率,又叫查全率，真实为正的样本中，预测为正的样本的比例

需要设置一个置信度阈值，大于这个阈值的才认为是正样本

![](assets/2023-09-11-09-02-16.png)

Map是综合不同阈值下的精度和召回率，得到的一个曲线下面积

![](assets/2023-09-11-09-05-40.png)

## YOLOv1

### 网络结构

![](assets/2023-09-11-09-07-34.png)

![](assets/2023-09-11-09-18-00.png)

![](assets/2023-09-11-09-24-37.png)

每一个特征点对应三十个值，5+5+20，5是c,x,y,w,h,20是类别，有两个锚框。

这两个五分别对应两个锚框，x,y是归一化后的位置，**w，h是相对于对应锚框的缩放比例**，回归要得到的，就是这30个值。



对于不同的值，就会使用**不同的损失函数**，比如x,y,w,h就是用MSE，类别就是用交叉熵。这样计算机才能知道，你这个值是什么意义。

![](assets/2023-09-11-09-46-45.png)

### 预测阶段

![](assets/2023-09-11-13-13-15.png)

![](assets/2023-09-11-13-13-37.png)

#### NMS

![](assets/2023-09-11-13-29-52.png)

如果想加强NMS，可以把阈值设置的低一些，这样更容易去掉重合的框。

![](assets/2023-09-11-13-34-12.png)

![](assets/2023-09-11-13-57-28.png)

### 训练阶段

![](assets/2023-09-11-14-06-42.png)