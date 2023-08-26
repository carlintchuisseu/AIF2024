# Recommendation Systems

## Course:
<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/SidOKu8RNmM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->

*   [Slides recommender systems](https://github.com/DavidBert/N7-techno-IA/raw/master/slides/Recommendation_System.pdf)  


## Practical session:
*   IMDB recommender system: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidBert/N7-techno-IA/blob/master/code/recommender_systems/INSA_Reco_TP.ipynb#scrollTo=BRuXLAqsabjZ)

*   Solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DavidBert/N7-techno-IA/blob/master/code/recommender_systems/INSA_Reco_solution.ipynb#scrollTo=y5KJkgtCZjH4&uniqifier=1)

## Project:

During the practical session, you saw how to build a recommender system based on content using the movie posters.  
Use `Gradio` to build a web app that takes as input a movie poster and returns the images of the 5 most similar movies according to their poster.  
The web app should be light and fast. Use a pre-trained network only to extract the vector representation of the input image and use the annoy index you built during the practical session to find the 5 most similar movies.    
Once this is done create a docker file to deploy your application.  Do not forget to include everything needed to run your application in the docker file.
Your app should look like something like this:
