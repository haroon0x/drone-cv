978-1-6654-9535-6/22/$31.00 ©2022 IEEE
Real-time person detection from UAV images using
performant neural networks
Alexandru Gabriel Popa
Faculty of Automatic Control and
Computer Science
University POLITEHNICA of
Bucharest
Bucharest, Romania
alexandru.popa0911@stud.acs.upb.ro
Loretta Ichim
Faculty of Automatic Control and
Computer Science
University POLITEHNICA of
Bucharest
Bucharest, Romania
loretta.ichim@upb.ro
Dan Popescu
Faculty of Automatic Control and
Computer Science
University POLITEHNICA of
Bucharest
Bucharest, Romania
dan.popescu@upb.ro
Abstract—Detection of persons from restricted areas with
the help of UAVs is one of the current trends in ensuring the
security of regions of interest. For the processing of images
acquired by UAVs of special interest are machine learning
techniques with an emphasis on neural networks. For this task,
two neural networks are proposed and tested in this paper:
YOLO v5m and Faster R-CNN. They were learned and tested
on two data sets: HERIDAL and a proprietary video stream
data set. Both networks gave good results, and the best results
were obtained on their own data set (because the images in
HERIDAL were taken at a higher height).
Keywords—person detection, image processing, neural
networks, UAV, statistic indicators
I. INTRODUCTION
Detection of persons from UAV images is a technique
widely used in search and rescue missions for lost or injured
people as well as in actions to monitor prohibited areas.
Detection of unauthorized persons and vehicles in restricted
areas (e.g., borders or prohibited areas) is a growing need in
the current security context (pandemic, conflict situations).
Due to current technological progress, the use of UAVs to
monitor these situations, coupled with advanced artificial
intelligence techniques for detecting suspicious people or
objects, is the best solution in terms of both cost and
performance. New trends in detecting objects (people, cars,
suspicious objects) in forbidden terrestrial areas are the use of
single or combined efficient neural networks in multi-network
systems. To improve statistical detection and time
performance, networks rely mainly on transfer learning. For
example, in [1] the authors proposed two methods used by the
Croatian mountain rescue service to detect people using
drones equipped with cameras/video cameras. The images
captured by them are analyzed on two fronts. One is
represented by two software applications, and the other by
human operators. The applications developed by them get a
better recall than the experts but lower accuracy.
Using other methods of people and vehicle detection and
tracking, such as satellites, manned helicopters, or ground
cameras, does not lead to better results because of their
resolution, high cost, or fixed position. Because the proposed
application relates to restricted area monitoring or rescue
operations, the broad ethical aspects of monitoring and
surveillance have not been considered.
The main goal of this paper was to implement real-time
person detection and tracking solutions using neural networks.
For this purpose, several neural networks have been
implemented and tested, focusing on two of the most efficient
ones, both in terms of statistical indicators and execution time,
YOLO-v5m and Faster R_CNN. The method developed for
people detection from UAV images using neural networks can
be successfully extended to search and rescue actions in case
of disasters.
II. RELATED WORKS
The tendency in person detection is to use neural networks
trained for object detection and classification. The authors in
[1] detect people in two stages. Initially, an attention-based
algorithm is used to detect regions of interest in images. In the
second stage, these segments are extracted and analyzed
(classified) by a convolutional neural network that determines
the areas that have the highest chances of containing people.
The obtained performances are good: 88.9% recall and 34.8%
precision. In parallel, an attempt was made to develop an
algorithm based on Faster R-CNN, but this network has
difficulties in detecting small objects. The obtained
performances are also good: 85.0% recall and 58,1%
precision. Also in this paper, the authors compare the
efficiency of the proposed algorithms with the efficiency of a
human operator certified for this type of search. On average,
an operator needs between 5 and 40 seconds to analyze an
image. Digital image processing can be done in less than a
second, and if a GPU is used then many images can be
analyzed in a very short time. The operator has a major
advantage when performing searches in an environment that
was not present at the training phase of the algorithms.
In [2] the authors used thermal cameras attached to drones
to detect human body heat, thus facilitating the search process.
However, the method is useful only in certain areas of the
globe. For example, in summer in some areas, the earth may
have a temperature like a human body, which is why the
method can no longer be used.
A multi-model-based system was proposed in [3] for
detecting people from aerial images in search and rescue
applications. The system is based on two convolutional neural
network architectures, one for the classification of the region
of interest and the second for classification. It was important
to consider the particularities of the areas surrounding the
framed region of interest, similarly to a search made by a
human operator. Learning and testing were based on the
HERIDAL database.
The main purpose of the authors in [4] was to detect people
in search and rescue applications from UAV images in
2022 14th International Conference on Electronics, Computers and Artificial Intelligence (ECAI) | 978-1-6654-9535-6/22/$31.00 ©2022 IEEE | DOI: 10.1109/ECAI54874.2022.9847477
Authorized licensed use limited to: J.R.D. Tata Memorial Library Indian Institute of Science Bengaluru. Downloaded on July 05,2025 at 17:22:43 UTC from IEEE Xplore. Restrictions apply.
mountainous regions. Here the target objects were small
compared to the acquired images which was a challenge. They
proposed reducing the search area by eliminating irrelevant
regions. For this purpose, they used a state-of-the-art neural
network, Faster R-CNN, to obtain a detection rate of 88.9%
and an accuracy of 34.8%. The database used was also
HERIDAL.
The method in which the person is classified using only a
small part of the image, the one with the highest probability of
finding a person, is good but it is the opposite of the methods
used by the human eye. When people analyze an image, they
also use its context, they use shadows, they use the type of
terrain, the height, etc. In the case of low-resolution images,
this context can provide more information compared to the
object itself searched [5].
III. MATERIALS AND METHODS
A. Neural Networks Used
YOLO (You Only Look Once) is one of the most popular
algorithms for detecting objects in images. For this task, it is
enough for it to be traversed only once. The first version of the
algorithm was published in 2015 by Joseph Redmon [6]. A
single neural network is used to determine both the framing
boxes of objects and to assign them the probability of
belonging to a certain class, in a single scrolling of the image.
This property ensures an increased processing speed so that
the detection and classification of objects in the video stream
can be done in real-time.
According to [6], the YOLO network is simpler in terms
of architecture. It directly uses the pixels of the image to obtain
the framing boxes and the probabilities of belonging to the
class. Thus, it is much easier to be optimized compared to
other object detection methods (such as Faster R-CNN).
Because the image is completely analyzed, YOLO has a great
advantage. Thus, the algorithm also learns information about
the context in which the objects are found. Compared to Faster
R-CNN, YOLO makes fewer mistakes when it comes to
confusing the background of images with the objects it is
trying to detect.
The latest version is YOLO v5 [11]. We used YOLO v5m.
The last two versions (YOLO v4 and YOLO v5) were
developed almost in parallel, the biggest difference being the
way the networks are implemented. YOLO v4 [10] stays in a
C-based implementation, while YOLO v5 moves to a Pythonbased implementation, using the Pythorch library. Initially,
there were no major differences in performance, but because
YOLOv5 is open source, it was improved by the community
and exceeded the performance of the YOLOv4 algorithm.
From the point of view of architecture, they are very similar
[12]. In both cases, it is made up of 3 parts. The first part is
responsible for extracting features from images, it is based on
an improved version of the network developed in previous
versions, Darknet. The new version is called CSPDarknet,
where CSP comes from Cross Stage Partial Network (a
network that works like DenseNet, which transfers one copy
of features from one layer to the next through a densely
connected block). YOLOv5 is differentiated by the fact that it
also uses a Focus block at the entrance of the first part (Fig.
1).
The next part aims to combine the extracted features, from
several previous stages, and obtain 3 sets of features of
different sizes. This part is based on the principle of spatial
pyramid pooling (SPP) and the PANet (Path Aggregation
Network) aggregation method. The last part is the same as in
the case of the YOLOv3 [9] network, composed of 3 detectors
that apply to the 3 sets of features of different sizes in the
previous layer. The architecture can be seen in Fig. 1, where
the CBL blocks are convolution blocks. Their structure is
more complex, it is not a simple convolution layer, they
contain several layers of convolution, and normalization and
are activated by the Leaky ReLu function. C3 blocks are
complex blocks, which also contain convolution layers.
Fig. 1. YOLOv5 architecture, inspired by [12] and [13].
The R-CNN network [7] consists of a three-module
system. The first module proposes the regions of interest.
These regions will be analyzed by the following two modules.
The second module is a convolutional neural network that
extracts a set of features (fixed-length vector) from each
previously proposed region. The last module is represented by
a set of SVM (Support vector machine) classifiers, one for
each class (trained by the one-versus-rest method). This latest
version of the algorithm is called Faster R-CNN and promises
to solve the problem of the long time required to propose areas
of interest. To obtain a short time frame for proposing areas of
interest, a special convolutional network of the proposal of
regions of interest (RPN - Region Proposal Network) has been
introduced [7]. The new method of proposing the regions
obtains the candidates in 10 ms for each image. RPN predicts
both the framing boxes of objects and a score that represents
the probability that there is an object in that box. By
combining the RPN network (the first module) and the Fast RCNN network, a new network is obtained. Both have in
common the same characteristics (Fig. 2).
Fig. 2. Faster R-CNN Network Architecture [7].
Anchor boxes are the most important concept of the Faster
R-CNN method because they are responsible for providing
default frames of different sizes and proportions. These can be
changed during training to suit the objects in the data set. The
original implementation contained 9 such frames, varying in
Authorized licensed use limited to: J.R.D. Tata Memorial Library Indian Institute of Science Bengaluru. Downloaded on July 05,2025 at 17:22:43 UTC from IEEE Xplore. Restrictions apply.
size and proportions. Because these borders overlap, an
algorithm called Non-Maximum Suppression is used to decide
which border to keep.
B. Dataset Used
We tested the proposed neural networks on two different
datasets. The first dataset used was HERIDAL [7]. This
dataset was developed by Split University in Croatia and aims
to develop the field of image processing for search and rescue
missions. The dataset contains approximately 1500 highresolution network images (4000 x 3000 pixels) that can be
used for training and 101 images that can be used for testing.
These images were captured from the drone from a high (but
varied) height. The images contain people in different poses
such as: lying down, squatting or walking. There are some
images that do not contain people. The environment in which
these pictures were taken differs greatly. For example, some
of the images were captured in the forest, where the vegetation
is dense and the visibility on the ground is reduced, people can
only be seen among the branches of the trees. Other images
were captured in the plains, in the park, or in a rocky
environment, on a mountain top. The data set contains images
from all seasons. Therefore, the neural network tasks were
very difficult. Some examples are given in Fig. 3 as ground
truth.
Fig. 3. Examples of images with persons in the HERIDAL dataset, used as
ground truth.
From the point of view of the second data set (own
dataset), it was obtained starting from a video intended for
tracking people by the drone. The persons were tracked by the
drone from a height of 30-50m. The person being chased by
the drone is moving, also the drone is moving trying to follow
its path closely. The video was divided into images, the
sampling was done every second of the video, so images are
obtained in which people are in different poses.
The resulting data set was subsequently annotated using
the free LabelImg software. This software allowed us to draw
the framing frame and tag the main people in the image,
excluding people at the distance who appear episodically in
the video. The software allowed the generation of XML files
in the form of a specific format that contains the information
previously obtained: the location of the frame of the object and
the corresponding label. To be able to train both the YOLO v5
algorithm and the Faster R-CNN algorithm, certain processing
was required on these files, finally obtaining text files that
contain the information in the format accepted by each
network.
Thus, the final data set contains 184 images, 128
belonging to the training data set, and 56 belonging to the test
data set. Some examples are given in Fig. 4.
Fig. 4. Examples of images with persons in the own dataset used for
learning.
The reference score used for object detection is mAP
(mean Average Precision). To calculate this score we used
precision and recall, but the precision and recall in the case of
object detection are calculated based on IoU and a certain
threshold. Depending on this threshold and the IoU, it is
determined whether the prediction is considered true positive
or false negative. Then, based on the previous result, the
precision and recall are determined. The graph between the
two is drawn (precision and recall), and the area below it will
be the AP (average precision). Each class will have an
average accuracy, and the mean of all average precisions is
the mAP score.
IV. EXPERIMENTAL RESULTS AND DISCUSSIONS
The YOLO v5m algorithm has been trained for 200 epochs
using an Nvidia K80 video card. The training time was about
100 minutes. After only 25 epochs, the recall and accuracy
metrics reach a maximum of 1 on the validation set images
(which are the same as the test data set).
The mAP score is also used to analyze the accuracy of the
framing frames, which is an average of the IoU scores
described above. The mAP score calculated for a threshold
value of the IoU score of 0.5 finally reaches the value of 0.995,
the frame predicted by the algorithm being therefore very
close to the frame made manually. For better analysis, an
average of several mAP scores with thresholds between 0.5
and 0.95 with a threshold of 0.05 is also used. This score is
lower, with a value of 0.902, but the value is very good.
Some examples of predictions on the HERIDAL dataset
by the YOLO v5m algorithm can be seen in Fig. 5.
For our dataset, some examples of predictions by the
YOLO v5m algorithm can be seen in Fig. 6. Persons in the
ground truth images (left) are framed with blue and persons in
the predicted images (right) are framed with red.
Authorized licensed use limited to: J.R.D. Tata Memorial Library Indian Institute of Science Bengaluru. Downloaded on July 05,2025 at 17:22:43 UTC from IEEE Xplore. Restrictions apply.
Fig. 5. Predictions made by YOLO v5m on the HERIDAL dataset.
Fig. 6. Predictions made by YOLOv5 m on own dataset.
It can be observed that the YOLO v5m algorithm manages
to detect the moving person in each image. The confidence
percentage with which he makes predictions is over 0.9, with
an average of around 0.95.
The Faster-RCNN algorithm was trained on the same
equipment as the one mentioned above, over a period of 100
epochs (the algorithm reaching convergence), the training
duration is approximately 40 minutes (transfer learning).
Some examples of predictions by the Faster-RCNN
algorithm on the HERIDAL dataset can be seen in Fig. 7.
Fig. 7. Predictions made by Faster R-CNN on the HERIDAL dataset.
Some examples of predictions by the Faster-RCNN
algorithm on the own dataset can be seen in Fig. 8. Persons in
the ground truth images (left) are framed with blue and
persons in the predicted images (right) are framed with red.
In the case of this network, the precision and the recall
reach the maximum value of 1 for the test data. The mAP score
with a threshold of 0.5 has an equally good value, equal to 1.
Instead, the mAP score with a variable threshold has lower
values, of 0.792, the algorithm has very low values for the
high thresholds of 0.9 and 0.95. In other words, it can be said
that the algorithm predicts the classification of the person with
an error of up to 20%. If the two high thresholds are excluded,
then the mAP score obtained has the value of 0.955.
Unlike YOLO v5m, Faster R-CNN has a much higher
confidence score, in most cases being equal to 1 (there are a
few exceptions). In terms of the correctness of the framing
frames, Faster has poorer results when it comes to a high
threshold. For example, we can analyze the images e and f,
both in Fig. 8 and in Fig. 6. It can be observed that Faster RCNN makes a slightly "lighter" framing, leaving a larger space
between the person's legs and the edge of the border, while
YOLO v5m manages to be closer to the person, being closer
to the ground truth.
In terms of time performance, Faster R-CNN performs
better than YOLO v5m both in the transfer learning phase (in
our case 40 min compared to 100 min) and in the operating
phase (37 ms compared to 41 ms). Obviously, the time
depends on the equipment used, but the ratio is kept in favor
of Faster R-CNN.
Table I summarizes the performances of the two networks
on own dataset.
Authorized licensed use limited to: J.R.D. Tata Memorial Library Indian Institute of Science Bengaluru. Downloaded on July 05,2025 at 17:22:43 UTC from IEEE Xplore. Restrictions apply.
TABLE I. NETWORK METRICS (OWN DATASET)
Indicator/
Network
Precision Recall mAP(0.5) mAP
YOLO v5m 1 1 0.995 0.903
Faster RCNN
1 1 1 0.792
Fig. 8. Predictions made by Faster R-CNN on own dataset.
V. CONCLUSIONS
In the case of the own data set, both networks managed to
correctly detect the people in the image. No one was omitted
and there was no case in which the networks detected parts of
the background as people. There were small framing errors,
but they would not influence the proper functioning of a
possible application made with the help of these two networks.
The results obtained exceed those in the literature, but the
diversity of people and background elements must also be
considered. The data set used in this article maintains a
constant size of the people employed, which facilitates the
learning process. At the same time, the background elements
are not diversified. In the case of the HERIDAL dataset, the
performances were poorer because sometimes the persons in
images are difficult to identify even by human operators. As
the feature work, we will implement a collaborative multi
neural network-based system to identify objects (including
persons) from images at higher distances.
ACKNOWLEDGMENT
Project from the Operational Program Competitiveness,
Project code 121596, Innovative System for Cross-Border
Combating Terrorism, Organized Crime, Illegal Trafficking in
Goods and Persons.
REFERENCES
[1] S. Gotovac, D. Zelenika, Ž. Marušić, and D. Božić-Štulić, “Visualbased person detection for search-and-rescue with UAS: Humans vs.
machine learning algorithm,” Remote Sensing, vol. 12, 3295, 2020.
[2] Ž. Marušić, D. Božić-Štulić, S. Gotovac, and Tonćo Marušić, “Region
proposal approach for human detection on aerial imagery,” 3rd
International Conference on Smart and Sustainable Technologies
(SpliTech), pp. 1–6. 2018.
[3] M. Kundid Vasić and V. Papić, “Multimodel deep learning for person
detection in aerial images,” Electronics, vol. 9, 1459, 2020.
[4] D. Božić-Štulić, Ž. Marušić, and S. Gotovac. “Deep learning approach
in aerial imagery for supporting land search and rescue missions,”
International Journal of Computer Vision, vol. 127, pp. 1256–1278,
2019.
[5] A. Torralba, “Contextual priming for object detection,” International
Journal of Computer Vision, vol. 53, pp.169–191, 2003.
[6] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
once: Unified, real-time object detection,”
https://arxiv.org/abs/1506.02640, 2016.
[7] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards realtime object detection with region proposal networks,” Proceedings of
the 28th International Conference on Neural Information Processing
Systems, pp. 91–99, 2015.
[8] J. Redmon and A.Farhadi, “Yolo9000: Better, faster, stronger,” IEEE
Conference on Computer Vision and Pattern Recognition (CVPR),
2017.
[9] J. Redmon and A. Farhadi, “YOLOv3: An incremental improvement,”
https://arxiv.org/abs/1804.02767, 2018.
[10] A. Bochkovskiy, C.-Y. Wang, and H.-Y. Mark Liao, “YOLOv4:
Optimal speed and accuracy of object detection,”
https://arxiv.org/abs/2004.10934, 2020.
[11] G. Jocher, “Ultralytics/yolov5: v6.0 - YOLOv5n ’Nano’ models,
Roboflow integration, TensorFlow export, OpenCV DNN support,”
https://github.com/ultralytics/yolov5/releases, 2022.
[12] U. Nepal and H. Eslamiat. “Comparing YOLOv3, YOLOv4 and
YOLOv5 for autonomous landing spot detection in faulty UAVs,”
Sensors, vol. 22, 464, 2022.
[13] L. Zhu, X. Geng, Z. Li, and C. Liu, “Improving YOLOv5 with
attentionmechanism for detecting boulders from planetary images,”
Remote Sensing, vol. 13, 3776, 2021.
[14] R Girshick, J Donahue, T Darrell, and J Malik, “Rich feature
hierarchies for accurate object detection and semantic segmentation,”
IEEE Conference on Computer Vision and Pattern Recognition, pp.
580–587, 2014.
[15] J. R. R. Uijlings, K. E. A. van de Sande, T. Gevers, and A. W. M.
Smeulders, “Selective search for object recognition,” International
Journal of Computer Vision, vol. 104, pp.154–171, 2013.
[16] R. Girshick, “Fast R-CNN,” In Proceedings of the IEEE International
Conference on Computer Vision, pp. 1440–1448, 2015.
[17] K. Simonyan and A. Zisserman. “Very deep convolutional networks
for large-scale image recognition,” International Conference on
Learning Representations (ICLR), pp. 1–14, 2015.
Authorized licensed use limited to: J.R.D. Tata Memorial Library Indian Institute of Science Bengaluru. Downloaded on July 05,2025 at 17:22:43 UTC from IEEE Xplore. Restrictions apply. 
