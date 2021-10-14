---
layout: default
title: Proposal
---

## 2.2 Summary of the Project (30 points)
We will be training an algorithm that, given an input sound file, will output a configuration of noteblocks in Minecraft that would best recreate the given sound. We intend for the bot to be able to then place the blocks in Minecraft to do so. This program could be used to transcribe recordings of music in Minecraft, and if extended, could even do so for existing instruments.

## 2.3 AI/ML Algorithms (10 points)
We plan to use an unsupervised algorithm that uses reinforcement learning with the difference between input and output sounds as the metric.

## 2.4 Evaluation Plan (30 points)
While the algorithm itself will output a configuration of blocks, these blocks will be able to be translated as sounds within the Minecraft game. As such, the metric used will be the similarity between the input sound and output sound, as determined by the cosine similarity between a vectorized representation of both sounds. Given that we could not find any existing programs that do what we intend to accomplish, we will be using the similarity of the input sound to a single noteblock tuned to the input’s pitch as our baseline. We expect our approach to have a cosine similarity value that is at least an order of magnitude higher than the baseline.

We will qualitatively analyze our project’s success primarily by ensuring that it will be able to produce more similar output sounds as the number of noteblocks it has access to increases. We will use sanity cases such as silence and simple multi-tonal sounds. Thankfully, the nature of this project makes “visualization” easy in the form of the auditory output that we will be able to perceive and judge perceptually. Our moonshot case will be the recreation of human speech, as we have seen algorithms capable of doing so on synthesizers before.

## 2.5 Appointment with the Instructor (15 points)
2:30pm - 2:45pm, Tuesday, October 19, 2021
