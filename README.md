
# SP-2 Dataset and RICAPS for Broadcast Sports video classification
You will find two things in this repository:

1. Material and details for SP-2 dataset
1. RICAPS implementation and code guide

A large number of video content is uploaded to video-sharing platforms such as YouTube, Facebook, and Youku. These videos belong to numerous categories such as sports, music, animations, and documentaries. Throughout these video classifications, sports videos are considered as one of the richest sources of entertainment. Sports enthusiasts want to keep themselves updated on the latest happenings. This desire has opened a challenging research direction known as sports video summarization. These highlights contain the most exciting segments of sports videos. Such highlights help sports enthusiasts to keep themselves updated in a short period. Bloggers and broadcasters spend a huge sum of time, money and human efforts on manual extraction of highlights from raw sports videos. Sports video highlight generation is a subclass of video summarization which may be viewed as a subclass of sports video analysis.


#### If you realize the need for the dataset, please skip till the horizontal rule.

#### Why a new dataset?
As we all know the Deep learning framework requires a huge amount of data for better performance. Somehow, the field for sports video analysis did not receive serious attention. To overcome this gap a dataset (SP-2) related to broadcast sports videos. This dataset is annotated with the type of sports, playfield scenarios and the camera views of the dataset.


The more unfortunate thing is that none of the researches categorizes broadcast sports video as a different class of videos i.e. every research considers amateur outdoor videos as and broadcast sports videos the same. Please mind that there is a huge difference between the nature of these videos. unfortunately, no dataset is publicly available. 

#### Previous datasets related to Sports
Yes, there are datasets related to sports such as Youtube 8M, but ask yourself “Do you want to work on 8 million videos?”, “How can you trust the annotation of 8M videos which were generated using user tags”. “How can people in collect the dataset where youtube is not accessible or internet is not reliable?”. I don’t know about others, but I don’t have the resources to work on this dataset.
Some may even say that UCF101 and HMDB51 have a sports category. Yes, it has a sports category. But ask yourself, does it cover the broadcast sports category?  Dose these datasets cover the internal game actions or views?
More specifically, none of these datasets differentiate between street sports and broadcasted sports game.


#### What is the difference between amateur sports video and professionally broadcast videos?
I won't explain amateur videos here; such videos usually have an “Egocentric vision” or “first-person vision”. See https://en.wikipedia.org/wiki/Egocentric_vision for details.


On the other hand, Broadcast sports videos capture the same point from multiple cameras at different angles and astonishingly none of the view stays for more than a couple of seconds. Scientifically speaking, there is no continuity within the frames. See the figure below and try to visualize it. If we closely look at the first row of the figure below, it can be noticed that the camera is switching, the zoom and pan levels are changing. In the second row, it can be observed that the camera is moving along with the players while zooming, even switching at some points.

![Selected frame samples from SP-2 dataset (first two rows) and UCF101](https://raw.githubusercontent.com/abdkhanstd/Sports2/master/Figures/cv.png)

#### Where did you gather the videos?
From the internet and TV channels. 

#### What can this dataset be used for?
It can be used for testing your ideas, training your deep learning models. Nevertheless, Applications are vast. I have used it for broadcast sports video classification. The proposed model can accurately detect the type of sports in real-world scenarios. The reason for classifying the sports video is that different sports have different ruleset and have different views and situations. RICAPS have the state-of-the-art accuracy, we opened a field and closed it.

**Hint for future work**: Using the sports category and playfield scenarios, automatic highlights can be generated.
_____________________________________________________________________________________________________
#### About the dataset
SP-2 dataset contains above 23,000 video clips of various durations. These video clips are extracted from full length broadcasted sports videos. The sports class, playfield scenario, and game actions are annotated accordingly (see table below for game actions). These videos belong to fourteen different categories of sports i.e. snooker, volleyball, ice hockey, basketball, baseball, rugby, tennis, handball, hockey, badminton, table tennis, cricket, football, and soccer. The following figure shows Some sample along with sports category, and playfield scenario/game action.

The videos in sports action categories have a minimum of 10 groups to 14 groups, where each group can consist of 150 videos (on average) of relevant sport game action. The videos from the same group may share some common features, such as similar background, similar viewpoint, etc. as they were cropped from the same season or same long duration video of sports.


![Some sample along with sports category, and playfield scenario/game action](https://raw.githubusercontent.com/abdkhanstd/Sports2/master/Figures/samples.png)

#### Download Videos
The videos in this dataset are approximate 10 Gigabytes in size with varying durations. These videos are shared via Microsoft OneDrive business account (other mirrors can be arranged on demand. Please refer to contact info.)
Videos can be downloaded from [here](https://stduestceducn-my.sharepoint.com/:f:/g/personal/201714060114_std_uestc_edu_cn/EgojmAd-CoxLqTWhDRFeN-kBH98O6d-hHhyTD8BM6KPH5A?e=MVe4AY)

#### Statistical details about the dataset
Here are some statistical details about the videos in the dataset.

|Class|Total groups |Total videos|Average videos/group|Total group duration|Average video duration |Game action classes|
| ------------- | ------------- | ------------- | ------------- | ------------- |  ------------- |  ------------- |
|Cricket|13|1773|136.4|9785.1|5.5|batting, bowling, run, out, event|
|Football|10|1613|161.3|11693.1|7.2|play, goal, foul|
|Soccer|14|1554|111.0|14254.3|9.2|play, goal, foul|
|Basketball|12|1790|149.2|14186.2|7.9|play, goal, foul|
|Baseball|10|1619|161.9|12063.7|7.5|batting, bowling, run, out, event|
|Rugby|10|1616|161.6|9346.3|5.8|play, goal, foul|
|Tennis|12|2062|171.8|11558.3|5.6|play, drop, service|
|Handball|11|1766|160.5|12468.0|7.1|play, goal, foul|
|Snooker|10|1376|137.6|8727.3|6.3|shot, pocket, aiming|
|Volleyball|10|1654|165.4|12944.2|7.8|play, drop, service|
|Ice hockey|10|1751|175.1|10510.1|6.0|play, goal, foul|
|hockey|10|1652|165.2|11080.1|6.7|play, goal, foul|
|badminton|13|1532|117.8|9333.5|6.1|play, drop, service|
|Table tennis|10|1267|126.7|7786.8|6.1|play, drop, service|


#### Test and Train List
Test and train lists can be found in the "List" folder. All of these lists were generated randomly. It is very important to keep the videos belonging to the same group separate in training and testing. Since the videos in a group are obtained from the same sports season and long duration sports video, sharing videos from same group in training and testing sets would give high performance and the trained model would not generalize properly.



#### How to run the code?
The training and testing codes can be found in the code folder. The code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the requirements.txt file. To ensure you're up to date, run:

`pip install -r requirements.txt`

Download the dataset and extract all the files without paths in the data folder. Next, create folders (still in the data folder) with mkdir train && mkdir test && mkdir sequences && mkdir checkpoints. You must also have `ffmpeg` installed in order to extract the video files. If `ffmpeg` isn't icluded in your system path (ie. which ffmpeg doesn't return its path, or you're on an OS other than *nix), you'll need to update the path to ffmpeg in `data/2_extract_files.py`

Before you can run `Train_IR_2.py`, you need to extract features from the images with the CNN. This is done by running:
` python extract_features_IR.py`
#### How to cite?
Will be added soon

#### Important note
I did not upload the palyfield and view annotations intentionally. More details (code + results) will be added later.


#### Acknowledgment
Thanks to Mr. Waqas Amin, Tahseen Khan, and other sports lovers who helped in, locating, cropping and annotating the videos.  Moreover, thanks to harvitronix for providing the base code.

