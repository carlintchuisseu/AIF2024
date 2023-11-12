# Recommendation Systems

## Course:

*   [Slides recommender systems](https://docs.google.com/presentation/d/1xUMmOn3vUXv8YgPmvTeXMLXDcSyOB9Sa8lLL2N2AXqk/edit?usp=sharing)  


<iframe width="560" height="315" src="https://www.youtube.com/embed/5TYgk0jZApc?si=NXGtORCgbxbQLJkj" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/Z5Vyx5AMnns?si=3Y7KOsqaoCSQRRjP" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/Hs-wURnnEwg?si=pqol8S2JlmY-lnrF" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/ztMvKeSOye8?si=k75YUUHzW7bpsAq-" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/eZHGfpP-FKQ?si=rTxfohLS737b1Bxq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/fiSHiAKxuNo?si=wfZVN3ZT75reBP_U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Practical session:
*   IMDB recommender system: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidBert/AIF2024/blob/main/rec_sys/recommender_systems/INSA_Reco.ipynb)

<!-- *   Solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidBert/AIF2024/blob/solutions/rec_sys/recommender_systems/INSA_Reco_solution.ipynb) -->

## Project:

During the practical session, you saw how to build a recommender system based on content using the movie posters.  
Use `Gradio` to build a web app that takes as input a movie poster and returns the images of the 5 most similar movies according to their poster.  
I would like you to mimic a real recommender system using a vector database.  To do so I want the database to be requested by the web app through a REST API. 
The web app should be light and fast. Use a pre-trained network only to extract the vector representation of the input image and call through the REST API the annoy index you built during the practical session to find the 5 most similar movies.     
Once this is done create a docker file to deploy your application.  Do not forget to include everything needed to run your application in the docker file.
