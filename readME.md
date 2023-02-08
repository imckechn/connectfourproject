# Training AlphaGO on Connect4
### A Project by Sam Vermeulen, Ian McKechnie, and Mihai Nica

## Introduction
Board games have long been a benchmark for progress in AI with IBM’s Deep Blue being the first AI to achieve Grandmaster level performance in Chess. The next significant achievement came in 2016 when Google DeepMind’s AlphaGo became the first AI to defeat a professional human Go player. DeepMind continued their work on AlphaGo and created AlphaZero which was able to achieve superhuman level performance in the Chess, Shogi, and Go with zero prior knowledge of the game. AlphaZero was trained by playing around 44 million games against itself and used around 5000 TPUs.

In this project, we implement a simplified version of the AlphaZero algorithm to play the game Connect Four that can run on a single GPU. Our agent uses upper confidence bound (UCB) action selection and guided random rollouts. We show that it can learn interesting strategies while searching a relatively low number of trajectories with zero prior knowledge.

## To Run locally

Run ```pip3 install -r requirements.txt``` to install all the required packages.

Then run ```$ pip install git+https://github.com/deepmind/dm-haiku``` to install the dm-haiku package.  This is a package that allows for the creation of neural networks in a functional style.

Then open the file ```connect_four_notebook.ipynb``` and run the cells.

## To run on Google Colab
Then open the file ```connect_four_notebook.ipynb``` and select the 'Open in Colab' button at the top of the page.  This will open the notebook in Google Colab.  Then run the cells.