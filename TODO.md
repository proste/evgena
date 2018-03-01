### Project structure
- add requirements file
- setup file? (and how/what)
- decide the need for tests, continuous integration
- top-level "project" emnist, possibly imagenet or so in future (downloading dataset)
- evgena module dataset - defining unified interface and implementing convenient functions for loading various datasets

### Guidelines
- learn to use logging (and try to actually use it)
- argparse
- setuptools for easy distribution (https://packaging.python.org/)

### The real stuff
- try to port PyGAA, understand what's going on
- create mating operators
    - one point (horizontal/vertical) crossover
    - two point (rectangular area) crossover
    - some blending crossover (ie kind of gaussian circle blend) - analogy of weighted arithmetic crossover 
- try to work with masked individuals - masked numpy arrays/sth different
- simple grid search based SVM trainer/the best possible models trained on the internet
- how to measure likeness? according to norm? part of loss/fitness, GAN
- add easter eggs

### GA
- change ordering of params in operations, so that sub ops first

### Utils
- keras callback plotting acc and loss
- progress bar

### First to do


### Img2Evgena
- evolution
	- evolving sequence of distortions of image
	- evolving noise to add

- trying to find whether noise or sequence correlates with source -> dest class pair

- needs of evolution
	- working with 3D/2D image data
	- image handling functions (imagemagick, openCV python API?)

### AlfaNum2Evgena
- models

### Unified downloader - some try_access_file, remote file repository links, hashes etc. refil.json, create packer, unpacker checker, add to starter notebook, later in install? some kind of install?
