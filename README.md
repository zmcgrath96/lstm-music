

# Says One Neuron to Another
Repository for EECS 738 Project Neural Network

#### Authors
Austin Irvine (https://github.com/austinirvine), Zachary McGrath, Ron Moore (https://github.com/ronmoore3), Matthew Taylor (https://github.com/mt3443), Kevin Ray (github.com/kray10)

### Data Sets
[Jazz MIDI dataset](https://www.kaggle.com/saikayala/jazz-ml-ready-midi)


[Classical music source](http://www.piano-midi.de/)

## Installation

Clone the repo
```
%> git clone https://github.com/zmcgrath96/lstm-music
```

Install dependencies
```
$HMM> pip3 install -r requirements.txt

```
## Theory
### Network Choice
The network is a Long Short-Term Memory (LSTM) neural network. LSTM is desireable as it a branch of Recurrent Neural Network (RNN). RNNs are good for time series data as they have a memory of the past events to help predict the next observation, unlike the classical architecture where the output is purely based on input at a certain time. The LSTM is an improvement in that it allows for longer lengths of time to be learned. For example, you could expect a RNN to read in 'The quick brown' and output 'fox'. If you gave this same RNN the string 'I am from Mexico. ... I speak fluent' you would hope it would output 'Spanish' but its not guaranteed. The LSTM has a sense of context and should be able to finish the above with 'Spanish'. 
## Running
## Process









