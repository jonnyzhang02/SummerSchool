Humans glance at an image and instantly know what objects are in the image, where they are, and how they interact. The human visual system is fast and accurate, allowing us to perform complex tasks like driving with little conscious thought. Fast, accurate algorithms for object detection would allow computers to drive cars without specialized sensors, enable assistive devices to convey real-time scene information to human users, and unlock the potential for general purpose, responsive robotic systems.

Current detection systems repurpose classiﬁers to perform detection. To detect an object, these systems take a classiﬁer for that object and evaluate it at various locations and scales in a test image. Systems like deformable parts models (DPM) use a sliding window approach where the classiﬁer is run at evenly spaced locations over the entire image.

YOLO is refreshingly simple: see Figure 1. A single convolutional network simultaneously predicts multiple bounding boxes and class probabilities for those boxes.YOLO trains on full images and directly optimizes detection performance.This uniﬁed model has several beneﬁts over traditional methods of object detection.

First, YOLO is extremely fast. Since we frame detection as a regression problem we don’t need a complex pipeline.We simply run our neural network on a new image at test time to predict detections. Our base network runs at 45 frames per second with no batch processing on a Titan X GPU and a fast version runs at more than 150 fps. This means we can process streaming video in real-time with less than 25 milliseconds of latency. Furthermore, YOLO achieves more than twice the mean average precision of other real-time systems. For a demo of our system running in real-time on a webcam please see our project webpage:http://pjreddie.com/yolo/.

Second, YOLO reasons globally about the image when making predictions. Unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance.Fast R-CNN, a top detection method [14], mistakes background patches in an image for objects because it can’t see the larger context. YOLO makes less than half the number of background errors compared to Fast R-CNN.

Third, YOLO learns generalizable representations of objects. When trained on natural images and tested on artwork, YOLO outperforms top detection methods like DPMand R-CNN by a wide margin. Since YOLO is highly generalizable it is less likely to break down when applied to new domains or unexpected inputs.YOLO still lags behind state-of-the-art detection systems in accuracy. While it can quickly identify objects in images it struggles to precisely localize some objects, especially small ones. We examine these tradeoffs further in our experiments.

All of our training and testing code is open source. A variety of pretrained models are also available to download.

我们所有的训练和测试代码都是开源的。各种预训练模型也可供下载。

人类瞥一眼图像，就能立即知道图像中有哪些物体，它们在哪里，以及它们是如何相互作用的。人类的视觉系统快速而准确，使我们几乎不需要有意识的思考就能完成驾驶等复杂任务。快速、准确的物体检测算法可以让计算机在没有专门传感器的情况下驾驶汽车，让辅助设备向人类用户传递实时场景信息，并释放通用、反应灵敏的机器人系统的潜力。

当前的检测系统是利用分类器进行检测。要检测一个物体，这些系统需要使用该物体的分类器，并在测试图像的不同位置和尺度上对其进行评估。可变形部件模型 (DPM) 等系统使用滑动窗口方法，在整个图像上均匀分布的位置运行分类器。

YOLO 简单得令人耳目一新：见图 1。单个卷积网络可同时预测多个边界框和这些边界框的类别概率。YOLO 在完整图像上进行训练，并直接优化检测性能。

首先，YOLO 速度极快。我们只需在测试时在新图像上运行神经网络，即可预测检测结果。我们的基础网络在 Titan X GPU 上以每秒 45 帧的速度运行，无需批处理，而快速版本的运行速度则超过了每秒 150 帧。这意味着我们可以实时处理流媒体视频，延迟时间不到 25 毫秒。此外，YOLO 的平均精度是其他实时系统的两倍多。有关我们系统在网络摄像头上实时运行的演示，请参阅我们的项目网页：http://pjreddie.com/yolo/。

其次，YOLO 在进行预测时对图像进行全局推理。与基于滑动窗口和区域建议的技术不同，YOLO 在训练和测试时能看到整个图像，因此它能隐含地编码有关类别及其外观的上下文信息。与快速 R-CNN 相比，YOLO 的背景错误率不到一半。

第三，YOLO 可学习对象的通用表征。在自然图像上进行训练并在艺术品上进行测试时，YOLO 的表现远远超过 DPM 和 R-CNN 等顶级检测方法。由于 YOLO 具有很强的通用性，因此在应用于新领域或意外输入时，不容易出现问题。虽然 YOLO 可以快速识别图像中的物体，但在精确定位某些物体，尤其是小物体方面却很吃力。我们将在实验中进一步研究这些权衡问题。


We unify the separate components of object detection into a single neural network. Our network uses features from the entire image to predict each bounding box. It alsopredicts all bounding boxes across all classes for an image simultaneously. This means our network reasons globally about the full image and all the objects in the image.The YOLO design enables end-to-end training and real-time speeds while maintaining high average precision.

Our system divides the input image into an S × S grid.If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

Each grid cell predicts B bounding boxes and conﬁdence scores for those boxes. These conﬁdence scores reﬂect how conﬁdent the model is that the box contains an object and also how accurate it thinks the box is that it predicts. Formally we deﬁne conﬁdence as Pr(Object) ∗ IOUtruthpred . If no object exists in that cell, the conﬁdence scores should be zero. Otherwise we want the conﬁdence score to equal the intersection over union (IOU) between the predicted box and the ground truth.

Each bounding box consists of 5 predictions: x, y, w, h, and conﬁdence. The (x, y) coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the conﬁdence prediction represents the IOU between the predicted box and any ground truth box.

Each grid cell also predicts C conditional class probabilities, Pr(Classi|Object). These probabilities are conditioned on the grid cell containing an object. We only predict one set of class probabilities per grid cell, regardless of the number of boxes B.

At test time we multiply the conditional class probabilities and the individual box conﬁdence predictions,

Pr(Classi|Object) ∗ Pr(Object) ∗ IOUtruthpred = Pr(Classi) ∗ IOUtruthpred

which gives us class-speciﬁc conﬁdence scores for eachbox. These scores encode both the probability of that class appearing in the box and how well the predicted box ﬁts the object.



2.1. Network Design

We implement this model as a convolutional neural network and evaluate it on the PASCAL VOC detection dataset. The initial convolutional layers of the network extract features from the image while the fully connected layerspredict the output probabilities and coordinates.

Our network architecture is inspired by the GoogLeNet model for image classification. Our network has 24 convolutional layers followed by 2 fully connected layers.Instead of the inception modules used by GoogLeNet, we simply use 1 × 1 reduction layers followed by 3 × 3 convolutional layers, similar to Lin et al . The full network is shown in Figure 3.

We also train a fast version of YOLO designed to push the boundaries of fast object detection. Fast YOLO uses a neural network with fewer convolutional layers (9 instead of 24) and fewer filters in those layers. Other than the size of the network, all training and testing parameters are the same between YOLO and Fast YOLO.



Training

We pretrain our convolutional layers on the ImageNet 1000-class competition dataset [30]. For pretraining we use the ﬁrst 20 convolutional layers from Figure 3 followed by a average-pooling layer and a fully connected layer. We train this network for approximately a week and achieve a single crop top-5 accuracy of 88% on the ImageNet 2012 validation set, comparable to the GoogLeNet models in Caffe’s Model Zoo [24]. We use the Darknet framework for allntraining and inference [26].

我们在 ImageNet 1000 级竞赛数据集 [30] 上对卷积层进行预训练。在预训练中，我们使用了图 3 中的前 20 个卷积层，然后是平均池化层和全连接层。我们对该网络进行了大约一周的训练，在 ImageNet 2012 验证集上的单作物前五名准确率达到 88%，与 Caffe Model Zoo [24] 中的 GoogLeNet 模型相当。我们使用 Darknet 框架进行所有训练和推理 [26]。

We then convert the model to perform detection. Ren et al. show that adding both convolutional and connected layers to pretrained networks can improve performance [29].Following their example, we add four convolutional layers and two fully connected layers with randomly initialized weights. Detection often requires ﬁne-grained visual information so we increase the input resolution of the network from 224 × 224 to 448 × 448.

然后，我们将模型转换为执行检测。Ren 等人的研究表明，在预训练网络中添加卷积层和连接层可以提高性能[29]。检测通常需要细粒度的视觉信息，因此我们将网络的输入分辨率从 224 × 224 提高到 448 × 448。

Our ﬁnal layer predicts both class probabilities and bounding box coordinates. We normalize the bounding box width and height by the image width and height so that they fall between 0 and 1. We parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1.

我们的最终层可预测类别概率和边界框坐标。我们根据图像的宽度和高度对边界框的宽度和高度进行归一化处理，使其介于 0 和 1 之间。我们将边界框的 x 坐标和 y 坐标参数化为特定网格单元位置的偏移量，使其也介于 0 和 1 之间。

We use a linear activation function for the ﬁnal layer and all other layers use the following leaky rectiﬁed linear activation:

We optimize for sum-squared error in the output of our model. We use sum-squared error because it is easy to optimize, however it does not perfectly align with our goal of maximizing average precision. It weights localization error equally with classiﬁcation error which may not be ideal.Also, in every image many grid cells do not contain any object. This pushes the “conﬁdence” scores of those cells towards zero, often overpowering the gradient from cells that do contain objects. This can lead to model instability,causing training to diverge early on.

我们在最后一层使用线性激活函数，所有其他层都使用leakyRELU激活函数：

我们对模型输出的**平方和误差**进行优化。我们使用平方总误差是因为它易于优化，但它与我们最大化平均精度的目标并不完全一致。它将定位误差与分类误差加权相等，这可能并不理想。此外，在每幅图像中，许多网格单元并不包含任何物体。**这使得这些单元格的 "可信度 "分数趋于零，往往会压倒包含物体的单元格的梯度**。这会导致模型的不稳定性，使训练在早期就出现偏离。

To remedy this, we increase the loss from bounding box coordinate predictions and decrease the loss from conﬁdence predictions for boxes that don’t contain objects. We use two parameters, λcoord and λnoobj to accomplish this. We set λcoord = 5 and λnoobj = .5.

Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reﬂect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly.

YOLO predicts multiple bounding boxes per grid cell.At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. This leads to specialization between the bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall.

为了解决这个问题，我们增加了边界框坐标预测的损失，并减少了不包含物体的边界框信念预测的损失。我们使用两个参数 λcoord 和 λnoobj 来实现这一目标。我们设置 λcoord = 5 和 λnoobj = .5。

总方误差也同样权衡大方框和小方框中的误差。我们的误差指标应该反映出，大方格中的小偏差比小方格中的小偏差更重要。为了部分解决这个问题，我们预测了边界框宽度和高度的平方根，而不是直接预测宽度和高度。

YOLO 为每个网格单元预测多个边框。在训练时，我们只希望每个对象由一个边框预测器负责。我们根据当前与地面实况的 IOU 值最高的预测结果，指定一个预测器 "负责 "预测一个物体。这就导致了边界框预测器之间的专业化。每个预测器都能更好地预测特定尺寸、长宽比或类别的物体，从而提高整体召回率。

During training we optimize the following, multi-part loss function: