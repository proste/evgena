# Evolutionary Generated Adversarial Examples #

Štěpán Procházka (author, [Charles University in Prague][mff])  
Roman Neruda (supervisor, [Academy of Sciences of the Czech Republic][avcr])  

[mff]: https://www.mff.cuni.cz/  
[avcr]: http://www.ustavinformatiky.cz/  

## Table of Contents ##

0.  [Introduction](#evolutionary-generated-adversarial-examples)
1.  [10 minutes to Evgena](#10-minutes-to-evgena)
2.  [100 minutes to Evgena](#100-minutes-to-evgena)
3.  [Features](#features)

## 10 minutes to Evgena ##
clone the repository

```shell
$ git clone https://github.com/proste/evgena.git
$ cd evgena
```

create environment, activate and install requirements

```shell
$ python3 -m venv .env
$ . .env
$ pip install -r requirements.txt
```

run jupyter notebook

```shell
$ jupyter notebook
```

## 100 minutes to Evgena ##


## Features ##
- datasets
	- fashion MNIST
	- MNIST
	- CIFAR-10
	- CIFAR-100
	- (imagenet)

- approaches
	- gradient based methods on target model (kind of baseline)
	- gradient based methods on surrogate model
	- GA methods
	- joint GA and gradient based methods on surrogate model

- tasks (each task with and without constraint on visual similarity)
	- given input example, modify to get desired class
	- given input examples, find universal modification to get desired class
	- given target class, generate corresponding input (inverse mapping)

- performance indicators
	- inference count on target model
	- (time/space) complexity
	- "aesthetics" of results
	- pros/cons in terms of constraints forced on environment (model, data)

