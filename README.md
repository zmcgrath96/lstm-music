

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
The initial trained set had a training accuracy of around 30%, with some in the 20s and some in the high 30s. Every LSTM had 2 layers of 256 nodes with one fully connected layer. The classes numbered about 9000 due to the unique combinations of notes and octaves. 

Its hard to judge the quality of music as it is very subjective. Making this process even more difficult is the sporadic nature of Jazz music. As a genre, jazz has lots of jumps in octaves, notes, instruments and many more. Without a thorough understanding of music theory or how to write music, we dove into this with an outside perspective. We had two rationales: music could either follow one 'root' instrument, or they could play off eachother. This is the reason for the three different architectures, as we wanted to see how each would perform. 
## Alternative Designs
Given the uniqueness of the 3 architectures, we will not dive into alternate architectures but instead how the LSTMs were changed in order to try for different results


