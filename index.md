---
title: Spotify Playlist Popularity
---

#### By Alexandra Abrahams and Kahunui Foster
Group #28
TF: Cindy Zhao

CS109A Introduction to Data Science, Fall 2017, Harvard University

# Final Project Report

#### Table of Contents:
* [Problem Statement and Motivation]()

### Problem Statement and Motivation

We aim to build a model that predicts the popularity of a Spotify playlist given information about the constituent songs of that playlist. Spotify is one of the world's largest music subscription companies, with about 100 million users. The majority of those do not pay a subscription, however, and the company profits from these users by inserting advertisements between tracks. Therefore the company has a business motive to create playlists which attract more users to them, since listeners to a playlist means more opportunities to show advertisements. On a personal level, using data science to predict something as innately human as what music we like just seemed like a very interesting challenge to us.

We use the number of followers of the playlist as our measure of the playlist's popularity, and in the data collection section we will explain what variables we used as predictors.

But no prediction system is perfect, and it is bound to make erroneous predictions. Our model could make two kinds of errors:
- a false positive (an unpopular playlist was falsely predicted to be popular)
- or a false negative (a popular playlist was falsely predicted to be unpopular)

Before beginning to build our models, we asked ourselves whether one kind of error was more palatable to us than the other, and if so which type. We reasoned that if our particular goal is to return popular playlists, then we want to be quite certain that if we return a playlist it actually is popular. This means that we want a high true positive rate. It also means that we mind less if we falsely classify a popular playlist as unpopular. We decided to make this our particular aim, within the overarching goal of having an accurate model across both types of errors.

An economic rationale for this aim is that since a company like Spotify might have to invest considerable resources to license songs from artists, if we advise the company to pay those licensing fees to bring the playlist to their users then we should be quite sure of recouping the cost from showing advertising to a lot of followers. It is less painful if we neglect to find a promising playlist that would have done well, than if we cause the company to lose a lot of money on licensing songs for a playlist that users rarely listen to.

Therefore given that we called a playlist popular, we would like the chances to be high that it actually is popular. Hence we seek to create a model that has a low false positive rate. We also want to balance this aim with a desire to return a diversity of playlists as popular; if we are only predicting a few playlists as popular then our model would not be very useful for Spotify executives creating playlists.

### Introduction and Description of Data

Description of relevant knowledge. Why is this problem important? Why is it challenging? Introduce the motivations for the project question and how that question was defined through preliminary EDA.

The data was collected using the following steps:
1. We used the [Spotipy API wrapper](http://spotipy.readthedocs.io/en/latest/) to collect the basic information of a set of about 1,700 public Spotify-owned playlists.
  An additional step was required here to get the full information of each of the above playlists, corresponding to the ‘full’ object rather than the simplified version with fewer fields. The full object gave access to a list of the tracks in that playlist, and data about the playlist such as the number of end followers. Some of these playlists failed to carry basic information, and the overall number was reduced.

2. For each of the above playlists, our program iterated through the list of tracks in that playlist and for each track collected [audio analysis information](https://developer.spotify.com/web-api/get-audio-features/) such as ‘danceability’. While looping through each track, the program also collected the genre information of the track’s artist. However, each artist has multiple genres, which Spotify has organized alphabetically rather than by relevance. Therefore all the genres were added to the track to impute it's genre.

3. Finally, the set of JSON objects was turned into a CSV, where each row corresponded to a playlist and the audio analysis information contained in each track was aggregated into average values for that playlist, such as average danceability. A playlist's genre was imputed from the set of genres available for that playlist's tracks, which was in turn derived from the track's artist. This is not an ideal process since an artist can work in a diverse area of genres, but the alternative was to work with a smaller dataset of about 800 playlists which were directly assigned categories by Spotify. When we evaluated the trade-off between more data or more accurate genre data, we chose the former.

The result of this process was about 1,500 playlists. Roughly 142 of these playlists had no tracks with any associated genre in their artist, and therefore they were dropped.


## Literature Review/Related Work

## Modeling Approach and Project Trajectory

## Results, Conclusions, and Future Work

Here is *emph* and **bold**.
