---
layout: default
title: Status
---

### Video:
https://youtu.be/2JXjUvanLcc

## Project Summary:

We will be training an algorithm that, given an input sound file, will output a configuration of note blocks in Minecraft that would best recreate the given sound. The sound files are expected to be short (< 1 second in length). Therefore, this current iteration of the project favors mimicking short sound bites, rather than complex audio. However, this program could be later extended to transcribe recordings of music through Minecraft audio, and if extended, could even do so for existing instruments.

## Approach:

We decided to approach this problem using an unsupervised neural network with backpropagation. Additionally, we ran into initial issues processing Minecraft audio due to the game using a single audio file and pitch shifting it for different notes, rather than having multiple audio files of a single instrument. Therefore, the first step was to extract each sound file from Minecraft and manually pitch shift it in order to have a variety of sound files to work with. After converting those files from .ogg to .wav, the next step was to then vectorize that sound in order to have data our neural network could process which took another python script. Our post-processed data was a number of vectors which contained 60 frequency buckets from 0 to 1480 hz (the range of Minecraft audio). The amplitude of each frequency buket at each point in time was summed before the data was normalized based on the max aggregate amplitude. We then took the square root of the data to remove any time and amplitude correlations and stored our final vectors in a single .csv file.

The next step was to create the network, which we did through basic backpropagation with around 200 passes of the data (epochs). The network is initialized with uniform weights and the input sounds are fed through. After each epoch, we evaluate the similarity between the estimated output and true output. Afterwards the weights are recalculated to reduce the error rate, and the process is repeated.

The final step was for our reward function to evaluate the success of our network’s predictions. This task was done by calculating the similarity between the input sound to mimic and the sound being generated. We relied on spatial distance using cosine for similarity between each output vector and calculated the overlap between the two. For each frequency bucket, we found the proportionality of each magnitude and applied it by prioritizing the larger of the two values. Finally, we added the weighted proportions and averaged the respective proportion maxes in order to normalize our data into a binary output in order to fully evaluate the similarity between inputs and outputs.


## Evaluation:

We used various evaluation techniques. First, we calculated errors in our code file. Apart from evaluating numerical errors.  We also utilized human intuition and tried listening very carefully  for the similarities between the input audio file and output audio file. We three humans have a decently  good understanding of similar two sounds. We needed mechanisms which were constant and not impacted by the environment or human biases. So we did a comparison by plotting input audio file and output audio file on plots and  spectrograms. Here with the help of examples we can see the similarities between our input and output.

<img width="412" alt="Spectrogram" src="https://user-images.githubusercontent.com/8118708/142148139-4cecbad3-0f04-4d0e-9b7c-b437c369679e.png">

<img width="583" alt="Frequency" src="https://user-images.githubusercontent.com/8118708/142148128-6f4583d6-4157-43bb-acd7-1305561d95e7.png">

We evaluated the differences between Input and output plots and spectrograms. We can clearly see the similarities here and essentially our output plots and spactograms are without the initial noise of input. 

## Remaining Goals and Challenges:

For the next few weeks, we plan on expanding both the length of input audio to be mimicked and the number of sounds we can combine. Currently, we are limited to using a single instrument (the Minecraft harp) for mimicking sounds, so expanding the number of sounds we’re able to manipulate would enable us to create more refined and more accurate outputs. Additionally, we are also limited to reproducing a single, sustained note, so complex musical pieces or even speech reproduction is outside the range of our AI. In the future, we would like to tackle this challenge by reproducing longer and more complex noises.

Given our experience so far, the main challenge we’ll be facing is expanding the AI to process more complex sounds. However, some solutions could entail the need for some change in frequency detection, or even a manually established splitting of the input data. Regardless of the difficulty, the main decision would be if there should be pre-processing performed on the input data to allow for one of these approaches, or if we should take another approach to creating training data that would allow for a more flexible application of our model.


## Resources Used:
https://minecraft.fandom.com/wiki/Tutorials/Sound_directory#Locating_specific_sound_files
Extract_sounds.py to find minecraft sounds

https://cloudconvert.com/
Convert .ogg to .wav

https://academo.org/demos/spectrum-analyzer/
Visualize our desired vectorized data

https://minecraft.fandom.com/wiki/Note_Block
Learn that frequencies range from 92 - 1480 hz

Scipy
Spatial calculations of vectors
Visualize spectrogram of sound samples

Numpy
Advanced vector calculations

Librosa
Frequency visualization

