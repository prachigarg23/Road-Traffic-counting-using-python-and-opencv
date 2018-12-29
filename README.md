# Road Traffic Counting 

## Abstract

In this project we tackle the problem of building intelligent computer vision systems to solve the problem of traffic monitoring. We use background subtraction method to detect and count the number of vehicles in a video. Along with background subtraction we use meanshift method to ensure that no vehicle is counted more than once. This is a simple method that uses image processing concepts without using machine learning algorithms that require training. These algorithms start yielding reliable results after initialisation on a few 100 frames. It is a cheap solution that requires openCV and python to run the code on a video of the traffic. This teaches us the importance and relevance of basic image processing and computer vision principles. 

## Introduction

With the conception of smart city transmuting cities into digital societies, making the life of its citizens easy in every way, Intelligent Transport Systems become the indispensable component among all. In any city mobility is a key concern; be it going to school, college and office or for any other purpose citizens use the transport system to travel within the city. Integrating an intelligent transport system in the city can save commuter time, help regulate law and order systematically, reduce congestion, improve travel and transit information, generate cost savings to motor carriers and emergencies operators, reduce detrimental environmental impacts and also reduce commuter stress and irritation. Hence it is an effort towards leveraging public safety and comfort. 

ITS technologies assist states, cities, and towns nationwide to meet the increasing demands on surface transportation systems. It enables the roads to be more accommodative of the ever growing population and handle a large number of routine commuters with ease and efficiency. The efficiency of an ITS system is mainly based on the performance and comprehensiveness of the vehicle detection technology.  Vehicle detection and tracking are an integral part of any vehicle detection technology, since it gathers all or part of the information that are used in an effective ITS. The ITS develops schemes and algorithms based on vehicular data on each street. This data must reflect upon the number of vehicles passing through the road or highway on an average hourly basis, periods of the year with increased traffic flow, potential reasons for the increase, types of vehicles passing through, the speeds at which they pass. Besides these, it should also monitor and should be able to identify any accidents or mishaps that have occurred on the road. 
Hence, the main parameters that can be evaluated through video based vehicle detection and tracking include count, speed, vehicle classification, queue lengths, volume/lane, lane changes, microscopic and macroscopic behaviors. The aim of this project is to implement a motion detection and tracking algorithm to count road traffic passing through the road as captured by a video camera. We shall employ OpenCV and Python to implement motion detection and tracking using background selection algorithm. 

## Main theme of the work

To accomplish the task of vehicle detection and tracking using low cost hardware and in computationally constrained environments, we need an algorithm that would (i) acquire a whole image from a video and temporary store it for further analysis, (ii) separate the background from the foreground, and (iii) keep track of each vehicle using a unique label. 

We follow the following procedure to extract an accurate count of the traffic:

1. Employ background subtraction algorithms for foreground detection.
2. Apply OpenCV image filters to get rid of noise and narrow gaps in the foreground objects 
3. Detect objects in the foreground map using contours and filter these objects based on a minimum size of objects that represent vehicles.
4. Use path information to ensure that each vehicle is counted only once by the system.

Now we shall take up each of the steps in the pipeline separately.

### Background Subtraction 

Background subtraction is a major preprocessing step in many vision-based applications. Technically, we use this to extract the moving foreground from static background.
We subtract the an image containing objects by an image of the background alone to get the foreground objects in the scene. When shadows are involved, this becomes slightly complicated as the shadows are also detected in the foreground. There are various types of background subtractors like MOG, MOG2, KNN and GMG. OpenCV has built in functions to implement all these. 

The main reason for employing background subtractor is that these classes are specifically built with video analysis in mind, which means that the OpenCV BackgroundSubtractor classes "learn" something about the environment with every frame. For example, with GMG, you can specify the number of frames used to initialize the video analysis, with the default being 120 (roughly 5 seconds with average cameras). The constant aspect about the BackgroundSubtractor classes is that they operate a comparison between frames and they store a history, which allows them to improve motion analysis results as time passes.In this project we are using 500 frames to initialize the video analysis. 
Further Background subtraction is important because of its ability to detect shadows .This is absolutely vital for an accurate reading of video frames; by detecting shadows, you can exclude shadow areas (by thresholding them) from the objects you detected, and concentrate on the real features. It also greatly reduces the unwanted "merging" of objects.

In this project, we use 2 background subtraction algorithms and compare performance of the 2 in counting vehicles: (i) Mixture of Gaussians (MOG2)  (ii) K Nearest Neighbours (KNN)

**Background subtraction using Mixture of Gaussian 2:**
1. It is a Gaussian Mixture-based Background/Foreground Segmentation Algorithm. One important feature of this algorithm is that it selects the appropriate number of gaussian distribution for each pixel.
2. It provides better adaptability to varying scenes due illumination changes etc.
3. We have an option of detecting shadows or not. It marks shadows in gray but detecting shadows decreases speed. 

**Background subtraction using KNN :**
1. The threshold on the squared distance between the pixel and the sample is used  to decide whether a pixel is close to a data sample.
2. K is the number of samples that need to be within Threshold in order to decide that that pixel is matching the kNN background model.

### Cleaning the foreground mask
A simple approach to improve foreground mask is to apply a few morphological transformations. We fill holes with closing operation, remove noise with opening operation and use dilation to merge adjacent blobs. 
Inspecting the masks, processed frames and the log file generated with filtering, we can see that after filtering, algorithm detects vehicles more reliably, and we have mitigated the issue of different parts of one vehicle being detected as separate objects.
 
### Object detection using contours 
After we have the filtered foreground mask, we find contours. We fix a minimum height and width for which the object is a valid vehicle. This size is fixed on inspection of the sizes of the cars. We draw rectangles around the objects using contours and filters those objects that do not fulfill validity criteria. We calculate the center (centroid) of each object using height, width and origin coordinates of the image. These centers are passed on for further processing. 

### Applying the meanshift algorithm
Background subtraction along with meanshift helps in accurate counting since meanshift keeps the track of vehicles that are already identified by the background subtraction method . Meanshift is an algorithm that tracks objects by finding the maximum density of a discrete sample of a probability function (in our case, a region of interest in an image) and recalculating it at the next frame, which gives the algorithm an indication of the direction in which the object has moved. 

To avoid detecting the same vehicle more than once in the same frame or in consecutive frames, we use a path tracking algorithm for each vehicle. In this algorithm, the goal is to track the path traced by each vehicle and use euclidean distance to calculate the path that each new point belongs to. After all points corresponding to each path have been found, we count the vehicle corresponding to that path as 1 and increment the vehicle count. The algorithm can be stated as follows:

- On first frame. we just add all points as new paths. This is the initialisation of paths.

- For each frame after the 1st frame ,we try to allocate the new points to existing paths based on certain criteria. This criteria depends on the length of the path. If any points are left, we define new paths for them. The newly detected points refer to the valid objects on each new frame. 

- Repeat, for each path

- Repeat for each new point

  - If length(path) == 1 (there is a single point in that path), for each path in the cache we are trying to find the point (centroid) from newly detected objects which will have the smallest Euclidean distance to the last point of the path. 
  - If length(path) > 1, predict new point on the path using last 2 points in path. Find min distance between predicted point and current point. 
  - Find and store point having minimum to the path along with the distance  

- End inner loop

- The point with minimal distance is added to the end of the current path and removed from the list of new points awaiting allocation. 

- If there was no match for the path in the current frame, don’t remove it. 

- End outer loop

- If some points left after this we add them as new paths.

- And also we limit the number of points in the path.

After all paths have been found, we need to efficiently count the vehicles that enter the exit zone. Just consider the last 2 points in each path and check the following:

- Last point is in exit zone
- Second last point is not in exit zone 
- Length of the path should be greater than the minimum path size specified 

If all above conditions meet, we increase the vehicular count by 1 

The pipeline saves the foreground masks of each frame and images with boxes over objects in the frames. For each frame, it gives us the date, time of processing, frame number and the number of vehicles detected up to that point.  

## Results

We implemented the algorithm on the given dataset be using two different background subtraction methods using  MOG2 and KNN.

**In Background subtraction using MOG2:**

Mask:

![alt text](https://github.com/prachigarg23/Road-Traffic-counting-using-python-and-opencv/blob/master/sample_results/MOG2_mask.png)

Processed Image:

![alt text](https://github.com/prachigarg23/Road-Traffic-counting-using-python-and-opencv/blob/master/sample_results/MOG2.png)

**In Background subtraction using KNN:**

Mask:

![alt text](https://github.com/prachigarg23/Road-Traffic-counting-using-python-and-opencv/blob/master/sample_results/KNN_mask.png)

Processed Image:

![alt text](https://github.com/prachigarg23/Road-Traffic-counting-using-python-and-opencv/blob/master/sample_results/KNN.png)

The number of vehicles observed in both the algorithms are equal to 4184 , but the time taken by knn is almost twice that is mog2 , hence mog2 is considered the better choice. Time taken by mog2 is almost equal to the length of the video.

Both MOG2 and KNN background subtraction can be used to get shadow less mask images.

When shadow is not removed following mask is obtained .

Mask:

![alt text](https://github.com/prachigarg23/Road-Traffic-counting-using-python-and-opencv/blob/master/sample_results/KNN_mask_wshadow.png)

Result of using mask with shadow is as shown below, if mask is not remove boundary box of images end up getting overlapped.

![alt text](https://github.com/prachigarg23/Road-Traffic-counting-using-python-and-opencv/blob/master/sample_results/KNN_wshadow.png)


## Discussions and conclusions

Out of the two algorithms used for background detection it is clear that MOG2 is better in comparison to KNN ( Lazy computation methods ) because MOG2 performs real time analysis as required for practical purposes whereas KNN does not.

Morphological filters are powerful in making images consistent and improving the performance of object detection and tracking. 


## References 

- https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515

- https://medium.com/machine-learning-world/tutorial-making-road-traffic-counting-app-based-on-computer-vision-and-opencv-166937911660

- https://docs.opencv.org/3.4/db/d5c/tutorial_py_bg_subtraction.html




