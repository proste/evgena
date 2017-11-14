### Project structure
- add requirements file
- setup file? (and how/what)
- decide the need for tests, continuous integration

### Guidelines
- learn to use logging (and try to actually use it)
- opt for argparse or click
- setuptools for easy distribution (https://packaging.python.org/)

### The real stuff
- try to port PyGAA, understand what's going on
- create mating operators
    - one point (horizontal/vertical) crossover
    - two point (rectangular area) crossover
- try to work with masked individuals - masked numpy arrays/sth different
- look for datasets MNIST, restrict just to those? find alphanumeric dataset
- simple grid search based SVM trainer/the best possible models trained on the internet
- how to measure likeness? according to norm? part of loss/fitness, GAN
- add easter eggs

### First to do
- train SVM on mnist
- reproduce pyGAA functions