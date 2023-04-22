# AIN2002-Project

## Specification and Dependencies
You can read the "requirements.txt" for information on used packages and libraries. All packages are easily installed with !pip install [package], if your machine can't finish installing after a fair amount of time, I would recommend adding -vv to pip so you can see what is causing the halt. Usually the problem is with caching, then you can add --no-cache-dir to pip which should solve the problem.

## Training Code
For each model used in the paper, there is an aptly named [model name].py. All models except MLP has fixed random states and they use the whole unordered dataset which should make their results easily reproducible. MLP models have saved weight files in the form of [the net type]w[hidden size]e[epoch number](public kaggle score).pt, batch size is always fixed to 128 after it's affect within the range of 16-512 seen to be near non-existant aside from training time and intermediate loss.
