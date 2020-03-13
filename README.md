# SP-2 Dataset and RICAPS for Sports video generation classification
A large number of video content is uploaded to video-sharing platforms such as YouTube, Facebook, and Youku. These videos belong to numerous categories such as sports, music, animations, and documentaries. Throughout these video classifications, sports videos are considered as one of the richest sources of entertainment. Sports enthusiasts want to keep themselves updated on the latest happenings. This desire has opened a challenging research direction known as sports video summarization. These highlights contain the most exciting segments of sports videos. Such highlights help sports enthusiasts to keep themselves updated in a short period. Bloggers and broadcasters spend a huge sum of time, money and human efforts on manual extraction of highlights from raw sports videos. Sports video highlight generation is a subclass of video summarization which may be viewed as a subclass of sports video analysis.


#### If you realize the need for the dataset, please skip till the horizontal rule.

#### Why a new dataset?
As we all know the Deep learning framework requires a huge amount of data for better performance. Somehow, the field for sports video analysis did not receive serious attention. To overcome this gap a dataset (SP-2) related to broadcast sports videos. This dataset is annotated with the type of sports, playfield scenarios and the camera views of the dataset.


The more unfortunate thing is that none of the researches categorizes broadcast sports video as a different class of videos i.e. every research considers amateur outdoor videos as and broadcast sports videos the same. Please mind that there is a huge difference between the nature of these videos. unfortunately, no dataset is publicly available. 

#### Previous datasets related to Sports
Yes, there are datasets related to sports such as Youtube 8M, but ask your self  “Do you want to work on 8 million videos?”, “ How can you trust the annotation of 8M videos which were generated using user tags”. “How can people in collect the dataset where youtube is not accessible or internet is not reliable?”. I don’t know about others, but I don’t have the resources to work on this dataset.
Some may even say that UCF101 and HMDB51 have a sports category. Yes, it has a sports category. But ask your self, does it cover the broadcast sports category?  Dose these datasets cover the internal game actions or views?


#### What is the difference between amateur sports video and professionally broadcast videos?
I won't explain amateur videos here, such videos usually have an “Egocentric vision” or “first-person vision”. See !https://en.wikipedia.org/wiki/Egocentric_vision for details.


On the other hand, Broadcast sports videos capture the same point from multiple cameras at different angles and astonishingly none of the view stays for more than a couple of seconds. Scientifically speaking, there is no continuity within the frames. See the figure below and try to visualize it.
#### Where did you gather the videos?
From the internet and TV channels. 

#### What can this dataset be used for?
It can be used for testing your ideas, training your deep learning models. Nevertheless, Applications are vast. I have used it for broadcast sports video classification. The proposed model can accurately detect the type of sports in real-world scenarios. The reason for classifying the sports video is that different sports have different ruleset and have different views and situations. RICAPS have the state-of-the-art accuracy, we opened a field and closed it.

**Hint for future work**: Using the sports category and playfield scenarios, automatic highlights can be generated.


It is recommended to find the answers for these questions, then proceed further
_____________________________________________________________________________________________________



