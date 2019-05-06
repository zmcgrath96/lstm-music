

# Music Generation
EECS 738 Semester Project

#### Authors
Austin Irvine (https://github.com/austinirvine)

Zachary McGrath (https://github.com/zmcgrath96)

Ron Moore (https://github.com/ronmoore3)

Kevin Ray (https://github.com/kray10)

Matthew Taylor (https://github.com/mt3443)

### Data Set
[Jazz MIDI dataset](https://www.kaggle.com/saikayala/jazz-ml-ready-midi)

## Installation

Clone the repo
```
%> git clone https://github.com/zmcgrath96/lstm-music
```

Install dependencies
```
$lstm-music> pip3 install -r requirements.txt
```

## Theory
### Network Choice
The network is a Long Short-Term Memory (LSTM) neural network. LSTM is desireable as it a branch of Recurrent Neural Network (RNN). RNNs are good for time series data as they have a memory of the past events to help predict the next observation, unlike the classical architecture where the output is purely based on input at a certain time. The LSTM is an improvement in that it allows for longer lengths of time to be learned. For example, you could expect a RNN to read in 'The quick brown' and output 'fox'. If you gave this same RNN the string 'I am from Mexico. ... I speak fluent' you would hope it would output 'Spanish' but its not guaranteed. The LSTM has a sense of context and should be able to finish the above with 'Spanish'. 

This project is unique in its architecture. Three different approaches were taken:
1. One LSTM for the most common instrument in a genre with two probablist models for two other instruments. The input to the first probablistic model is the output from the LSTM, which choses a random sample from the distribution of the relationship between the first instrument note and the second instrument. A similar process is done for the final model, using the output of the second as the input to the third.

2. Three LSTMs with the most common instrument being the first LSTM. The output of this feeds into the LSTM of the second most common instrument which produces notes based on the first instrument. Finally, the last LSTM uses the output of the second LSTM as input and bases its notes to play off of the second instrument.

3. Three LSTMs are used again, but a 'root' instrument is the basis for all three instruments. 

The data set used for trainin is Jazz. The most popular instruments for this dataset were as follows: Piano, Bass, Saxaphone. Below are diagrams for the three architectures described with the instruments in their respective positions. 

#### Architecture 1
![architecture 1](https://github.com/zmcgrath96/lstm-music/blob/master/images/Architecture%201.png)
#### Architecture 2
![architecture 2](https://github.com/zmcgrath96/lstm-music/blob/master/images/Architecture%202.png)
#### Architecture 3
![architecture 3](https://github.com/zmcgrath96/lstm-music/blob/master/images/Architecture%203.png)

## Running
### Training
```
python3 main.py -t=<instrument> -arch=<architecture number>
```
The instrument parameter can take values of 'piano', 'bass' or 'sax'. Architectures can be 1, 2, or 3.
### Generating Music
```
python3 main.py -g -arch=<architecture number>
```
The architecture number can be 1, 2, or 3. The output is saved as a MIDI (.mid) file. A quick google search and you can find a midi player to upload the song to play.
## Results
The final design was trained over 20 epochs, all with around 80% training accuracy. While not the most pleasing music, there is some  rhythm with the piano. We trained with 50 epochs, but the performance was only marginaly better. 

Its hard to judge the quality of music as it is very subjective. Making this process even more difficult is the sporadic nature of Jazz music. As a genre, jazz has lots of jumps in octaves, notes, instruments and many more. Without a thorough understanding of music theory or how to write music, we dove into this with an outside perspective. We had two rationales: music could either follow one 'root' instrument, or they could play off eachother. This is the reason for the three different architectures, as we wanted to see how each would perform. 

We also embedded 'start' and 'end' meta notes in order to start and end the song. This was also unique, but didn't seem to help much with at the end. 
### Design Issues
One of the largest issues that we ran into was the data set. Jazz MIDI data sets are few and far between. We found one set that had plenty of songs, but they all needed to have the 3 instruments. This grately reduced our data set. In adddition to that, after listening to several of the songs, they really weren't Jazz, but instrumental versions of various genres. 

The output for each architecture are saved in the 'songs' folder and can be played on online MIDI players.

Another issues we came across was class size. Initially, we tried to incorporate all chords and notes used in the training set, but this proved to be a futile pursuit. The class size grew to around 9000 different classes. Because of the number of rests in a song and the fact that there aren't different types of rests, the model initially only guess rests, and it was reflected in the training accuracy. We decided to then remove chords and only use combinations of notes. This reduced the class size to 2000, but still very large and yielded a similar result. Finally we reduced it to only the possible set of notes with octaves, which reduced the class size to 88. 

Data cleaning was also an issue. Given the nature of the input, we had to remove chords and multiple notes over a time stamp to reduce the class size. 

### Alternative Designs
Several different designs were considered. Giving the piano a 'left' and 'right' hand for notes and chords would have allowed us to play chords with notes. The overlap of notes in time had to be omitted to reduce the class size.  

There were many, many features we did not include that could be included to change the output type. This includes volume of the notes, tempo and harmonies. Given how we already tried to play music from several instruments off of eachother, we did not want to change too many variables. 

