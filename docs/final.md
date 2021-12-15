---
layout: default
title: Final Report
---

## Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/7Wep6ku_xUU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Project Summary:
We will be training a backpropagation neural network, which, given an input sound file, will output a configuration of note blocks in Minecraft that would best recreate the given sound. It will process an input sound file to generate a discrete frequency mapping, then output a vector representing which note blocks to play in order to replicate the input sound. In addition, it will also generate a preview of what that would sound like, by combining the sound files for the aforementioned note blocks and placing it into the output destination.

This problem, while expectedly trivial for the simple, single-note cases, becomes more complex for sounds such as chords, and non-tonal noises. While it would be theoretically possible to heuristically iterate through each possible note block combination, a machine learning algorithm would be much more suited towards such complex cases.

## Approaches:
### Baseline:
The baseline approach would involve attempting to replicate the sounds by ear. This approach would have the advantages of being about as accurate as the amount of effort the transcriber would be willing to put in, and would not require any training data, as human beings are already primed for this type of task.

This approach, while viable-- indeed, many people have already utilized this approach by creating “noteblock” parodies of popular songs within Minecraft and uploading their creations to youtube--, is very tedious, and falls apart especially for sounds that do not have easily identifiable tonal distinctions, or have too many different composite tones to identify. In addition, it is not extendable-- it requires manual labor for each individual note. Overall, the baseline, while possible, is tedious-- and therefore it is this menial labor that we are attempting to solve.

### Heuristic:
In the heuristic approach, a set of limitations are imposed upon the potential sound space (such as the maximum number of overlapping notes, the range of notes, etc.), and all possible combinations are generated for the resulting domain. Given that our problem has a theoretically ideal solution for each sound (in a given domain, there must exist some combination of note blocks such that the “difference” between the input and output signs is minimized), this solution would likely be the most accurate. In fact, this approach would not even require training data; given its heuristic nature, it would only need to be run once to generate and store the sound profiles, after which it would be able to compare any given sound to its database and return the one which fits best.

However, this approach has a few limitations: this algorithm would have to run in factorial time without coming up with some clever optimizations (as every possible combination of sounds must be considered), and any gains made in runtime would have to be traded off with the algorithm’s accuracy as its domain size shrinks. In addition, it would not be very modular; additions to the domain of possible noteblock sounds or increments to the maximum number of allowed overlapping blocks would necessitate a regeneration of the database without significant spatial optimizations (such as determining which sounds would be newly generated, and only generating those).

Machine Learning-- Discrete Frequency Mapping (Chosen Approach):
In this machine learning approach, sounds are mapped to a set of discrete frequency ranges:
```python
  def map_to_frequency_ranges(range_size, soundfile):
    range_aggregate_amplitudes = []
    maximum_aggregate_amplitude = 0
    spectrogram = gen_spectrogram(soundfile)
    range = 100
    index = 0
		for each slice in spectrogram:
			if slice.frequency > range:
				if range_aggregate_amplitudes[index] > maximum_aggregate_amplitude:
          maximum_aggregate_amplitude = range_aggregate_amplitudes[index]
          index++;
          range += range_size
      range_aggregate_amplitude[index] += slice.amplitude
  for each aggregate_amplitude in range_aggregate_amplitudes:
    aggregate_amplitude /= maximum_aggregate_amplitude
    aggregate_amplitude = sqrt(aggregate_amplitude)
  return range_aggregate_amplitudes
```
This process is applied first to each possible note block sound in the domain (the results of which are stored in a csv file for later comparison), then to the input sound and output sounds of the algorithm while it is being run. As such, the neural network itself inputs the sound’s representation as a list of amplitudes for set ranges of frequencies.

The neural network being used is a backpropagation neural network. However, given that our algorithm necessitates a kind of self-supervised learning (as opposed to pure supervised learning, which would require heuristically or manually generated “answers” to the training data, or to pure unsupervised learning, which would end up generating its own categories), we needed to modify the neural network. Given that existing libraries we searched through did not provide an option for fully customizing a neural network’s error calculation, we built a simple neural network from scratch, simulating each individual node. In doing so, we were able to write our own reward function, which calculated the similarity between the input and output sounds using the cosine similarity between the vectorized representations of their respective amplitudes of set frequency ranges:
```python
  def calculate_reward(input_vector, output):
    components = []
    for each element in the output:
      if the element is 1: add the corresponding soundfile
    output_sound = overlapped soundfiles in components
    output_vector = map_to_frequency_ranges(range_size, output_sound)
    return 1 - cosine_similarity(input_vector, output_vector)
```
Given that using this reward function alone resulted in the algorithm pursuing local maxima and heavily overfitting the data (which was expected, given that we had not yet provided it with an accurate metric for how similar the sounds should be expected to be), we implemented an error function that took the derivative of the reward function with respect to the algorithm’s iteration count; in other words, we trained the neural network to punish stagnation (with limitations to account for the accomplishment of the “true” maxima), and reward incremental improvements to the accuracy of its predictions.

Using the previous maximum accuracy for a given training data input as a baseline, we measured by how much the algorithm had either improved or worsened for predicting that input. Then, we generated a “solution” vector that differed from the predicted vector based on its performance. For example, if the output vector was [0, 1], and the algorithm had performed a little bit worse than it had previously, we generated a solution vector of [0.6, 0.4]. If the algorithm had performed much worse than it had previously, we generated a solution vector of [0.95, 0.05]. By using this solution vector as the supervision metric, we were able to backpropagate error into the neural network in correlation with the level of desired accuracy.
```python
def calc_solution_vector(input_vector, output, local_maxima, input_index):
		solution_vector = []
		score = calculate_reward(input_vector, output)
		if score is greater than the local maxima for that input:
			margin = normalized amount improved
			for each element in the output:
				set corresponding element in solution vector such that it is closer to 1 or 0 depending on how much better it performed
		else:
			margin = normalized amount worsened
			for each element in the output:
				set corresponding element in solution vector such that it is further from 1 or 0 depending on how much worse it performed
		return solution_vector
```
In addition, we also attempted another method of generating a solution vector, which would simply concur with the output if the result was greater than the average accuracy so far, and give the opposite output if the result was worse than the average accuracy so far. 

Using these functions, we simulated a neural network with 60 input nodes (representing 60 the aggregate amplitudes of 60 frequency ranges), 20 hidden nodes in one hidden layer, and 13 outputs (representing an octave of harp sounding note blocks in the C scale).

This approach was chosen, as we predicted that it would be the most effective for solving more complex sounds. It would take less time to train than it would to generate the possible sounds for the heuristic approach, and while we knew that we would be making sacrifices in terms of accuracy (as the algorithm has no way of knowing what the best number of different note blocks might be, and the matching would, by nature, be fuzzy), we determined that take less time to train. At the time of our decision, we were uncertain as to whether overfitting would be an issue, due to the theoretical nature of the algorithms we came up with in order to avoid it.


### Machine Learning-- Acoustic Fingerprinting:
This approach, which we only came up with after we had finished the majority of our project, was an alternative proposed against the previous machine learning approach, which used distinct frequency ranges. With acoustic fingerprinting, we intended to identify key identifying factors for a given input sound-- such as primary frequency range, amplitude, attack, fade, etc.)-- and generate a vector around those factors instead.

The primary advantage of this approach is that it is a more accurate mapping of the sound profiles. While we believed ourselves to be mapping sounds to their frequency ranges, we later discovered that what we were doing was more accurately a form of projection; the function was not entirely ontological: it was possible for two distinct sounds to have very similar frequency profiles.

However, the downside of this approach is that it is incredibly sensitive to changes in the sound, and has no trivial method of determining how highly each factor should be weighted; this could result in two sounds with similar volumes or attacks being chosen, despite the actual frequencies being much further apart. In addition, a sound profile would be length-dependent-- our frequency mapping, on the other hand, took the average amplitudes of each range, and normalized the data, resulting in a length-independent approach. As such, we revised our actual objective to reflect the tone-focused nature of our project, and determined that this approach, while it may capture the input sound’s profile more accurately from a holistic standpoint, would not be optimized for distinguishing individual tones, which is what we had originally set out to do.


## Evaluation: 
### Accurate Similarity Metric:
First, we verified that the metric we were using was sound; if the comparison algorithm we were using to determine the accuracy of the network’s predictions was faulty, then it would mean that the algorithm would be training itself on a flawed premise. As such, we had to ensure not only that our algorithm would correctly identify similar sounds as being similar, but also that it would not incorrectly identify unsimilar sounds as being similar as well.

While we originally ran into issues caused by a lack of normalization in the data causing sounds with broad stretches of “noise” to map more closely with other “noisy” sounds, we managed to implement a solution which normalized the data based of the maximum amplitudes, and compress “noisy” parts of the sound to near irrelevancy.

Upon implementing this solution, we were able to see that the algorithm would accurately state that two different notes, played by two different instruments, were similar, while two different notes, played on the same instrument, were not. This behavior is illustrated below:

<img width="814" alt="c-soundfile-test" src="https://user-images.githubusercontent.com/8118708/146108604-874aa068-9267-402d-94a2-d2c684df25c1.png">

<img width="816" alt="harp-soundfile-test" src="https://user-images.githubusercontent.com/8118708/146108648-225bd584-ce33-474d-8d26-cfb44c54b256.png">

As desired, the frequency profiles for the C note played on both the harp and the flute line up almost perfectly, and yield a similarity score of 0.8100, whereas the frequency profiles for the C and D notes played on the harp yield a similarity score of 0.086.


### Quantitative:
From a quantitative standpoint, we discovered that our algorithm was, in fact, capable of accomplishing what we had set it out to do, in the most objective of terms: given a sound’s mapping to the aggregate amplitudes of set frequency ranges, it would generate a combination of sound files from its given domain, which, when played together, would map to the most similar respective aggregate amplitudes of the same set frequency ranges. In other words, our algorithm was designed to recreate a given sound’s quantized frequency spectrogram using sounds that it was given access to.

This objective, we believe, was sufficiently accomplished. We established the cosine similarity between two different instruments playing two different notes as the baseline, and found that on average, the resulting score was a value of around 0.8 similarity (on a scale of 0 to 1, where a value of 1 implies that the profiles are identical). 

However, while conducting this analysis, we discovered that we had to work around an unexpected property of the relationship between the input sounds and our processed data: the mapping we did of each sound to the set frequency ranges was not ontological, as we originally assumed, but rather more of a projection; multiple combinations of sounds could result in the same frequency profile, resulting in an inability to accurately distinguish between tones. 

We managed to mitigate this problem by further normalizing the data to account for the breadth of the frequency ranges-- meaning, if a particular sound had a very short range of frequencies with relatively high amplitudes-- what we will henceforth refer to as the “active” range-- (say, 550 to 660 hertz), and was very clearly an E5 note, we wanted the algorithm to favor selecting a corresponding E5 note with a slightly different “active” range (such as 500 to 600 hertz), rather than an A5 note that had a very broad “active” range (for example, 60 to 1200 hertz) which happened to overlap with that of the E5 note. In doing so, we were able to produce accurate representations of notes and chords in the majority of cases.

We accomplished this by using the proportion of overlapping “active” range in each frequency and multiplying them together, and using that to scale the cosine similarity value. By doing so, we were able to produce an algorithm that would more precisely attempt to replicate the composite notes of a given sound. To illustrate this capability, the following is a graph showing the original and replicated frequency profiles of a simple piano C chord:

<img width="813" alt="c-chord-soundfile-test" src="https://user-images.githubusercontent.com/8118708/146108700-ca96222b-aaae-4cef-947e-8d5a6b1e74e5.png">

As expressed, while the frequencies do not exactly line up, the resulting sound is relatively similar to the original. Before we made this change, the algorithm had preferred to generate a sound which, for example, would combine all the notes in-between C and G, to produce a profile which encompassed-- rather than overlapped-- the desired frequencies. As it stands, our C chord replication had a similarity value of 0.708, a value we consider sufficient.


### Qualitative:
Our qualitative analysis was relatively straightforward: the human ear is sufficiently skilled at determining whether two sounds are similar-sounding even when there aren’t any clearly distinct notes to make out; we wanted to determine the qualitative success of our algorithm by seeing if the input and output sound files sounded similar to the human ear for sounds which were considerably “noisier” than simple notes.

For this analysis, we selected a few sounds that we felt would be difficult for a person to manually replicate in minecraft, such as an accordion, the sound of a coin falling, and the sound of a balloon popping. While testing these sounds, we discovered our limitation was not with our algorithm, but rather with the breadth of sounds that Minecraft note blocks could provide; many of these sounds consisted of so much noise that there were not really any combinations of note blocks that could create a sound that was similar enough.

As such, we stuck to analyzing the results for what we felt was a good middle ground; the accordion. The accordion, while it is capable of playing individual notes, has a very broad “active” frequency range, thanks to the fact that multiple different outlets in the physical instrument itself are active at any given time. However, we felt that since these outlets would still have to map to individual notes, it would be possible to construct a combination of note blocks that replicated which notes were active.

To our pleasant surprise, we discovered that our algorithm was, in fact, capable of replicating such a sound. We discovered that the resulting combination of note blocks did, in fact, sound quite similar to the input accordion sound, given that it was constructed using only the harp note blocks, and a visual representation of the frequency profiles confirmed our results:

<img width="814" alt="accordion-soundfile-test" src="https://user-images.githubusercontent.com/8118708/146108737-b2fd2336-0784-4c79-a42c-ea454dd66740.png">

With a similarity value of 0.8327, we determined that our project was a success. While we encountered many roadblocks along the way, and had to make significant adjustments to our original approach, we were ultimately able to train a neural network to accurately reproduce input sounds by selecting a combination of note blocks in Minecraft that would sound similar. Counter-intuitively, while our algorithm performs at a less than perfect level for individual notes (a phenomenon that we attribute to the fact that the network, just by the nature of its output space, will naturally produce more outputs that use multiple notes than outputs that only utilize single notes), it performed exceedingly well for complex sounds, such as accordions. We believe that given more time and a larger domain (such as by including other instrument note blocks), we would accurately be able to replicate even more complex sounds, such as human speech.


## References:
https://minecraft.fandom.com/wiki/Tutorials/Sound_directory#Locating_specific_sound_files (extract_sounds.py to find minecraft sounds)

https://cloudconvert.com/ (to convert .ogg to .wav)

https://academo.org/demos/spectrum-analyzer/ (to visualize what our vectorized data ought to look like)

https://minecraft.fandom.com/wiki/Note_Block (to learn that note block frequencies range from 92 - 1480 hz)

