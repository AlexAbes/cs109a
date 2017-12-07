---
title: Spotify Playlist Popularity
---

#### By Alexandra Abrahams and Kahunui Foster
Group #28
TF: Cindy Zhao

CS109A Introduction to Data Science, Fall 2017, Harvard University

#### Table of Contents:
* [Goals](/#goals)
* [Data Collection Process]()
* [Approach to Modeling]()
* [Conclusions]()

## Goals

We aim to build a model that predicts the popularity of a Spotify playlist given information about the constituent songs of that playlist. We use the number of followers of the playlist as our measure of the playlist's popularity, and in the data collection section we will explain what predictors we used as predictors.

But no prediction system is perfect, and it is bound to make erroneous predictions. Our model can make two kinds of errors:
- a false positive (an unpopular playlist was falsely predicted to be popular)
- or a false negative (a popular playlist was falsely predicted to be unpopular)

Before beginning to build our models, we asked ourselves whether one kind of error was more palatable to us than the other, and if so which type. We reasoned that if our particular goal is to return popular playlists, then we want to be quite certain that if we return a playlist it actually is popular. This means that we want a high true positive rate. It also means that we mind less if we falsely classify a popular playlist as unpopular. We decided to make this our particular aim, within the overarching goal of having an accurate model across both types of errors.

An economic rationale for this aim is that since a company like Spotify might have to invest considerable resources to license songs from artists, if we advise the company to pay those licensing fees to bring the playlist to their users then we should be quite sure of recouping the cost from showing advertising to a lot of followers. It is less painful if we neglect to find a promising playlist that would have done well, than if we cause the company to lose a lot of money on licensing songs for a playlist that users rarely listen to.

## Data Collection process

## Approach to Modeling

## Conclusions

Here is *emph* and **bold**.
