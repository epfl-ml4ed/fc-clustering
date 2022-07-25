# Identifying and Comparing Multi-Dimensional Student Profiles across Flipped Classrooms

This repository is the official implementation of the AIED 2022 Paper entitled "Identifying and Comparing Multi-Dimensional Student Profiles across Flipped Classrooms" written by Paola Mejia, Mirko Marras, Christian Giang and Tanja Käser.

Flipped classroom (FC) courses, where students complete pre-class activities before attending interactive face-to-face sessions, are becoming increasingly popular. However, many students lack the skills, resources, or motivation to effectively engage in pre-class activities. Profiling students based on their pre-class behavior is therefore fundamental for teaching staff to make better-informed decisions on the course design and provide personalized feedback. Existing student profiling techniques have mainly focused on one specific aspect of learning behavior and have limited their analysis to one FC course. In this paper, we propose a multi-step clustering approach to model student profiles based on pre-class behavior in FC in a multi-dimensional manner, focusing on student effort, consistency, regularity, proactivity, control, and assessment. We first cluster students separately for each behavioral dimension. Then, we perform another level of clustering to obtain multi-dimensional profiles. Experiments on three different FC courses show that our approach can identify educationally-relevant profiles regardless of the course topic and structure. Moreover, we observe significant academic performance differences between the profiles.  


## Usage Guide

The code has was developed with Python 3.8.8. We recommend creating a python virtual environment (to avoid conflicts with other packages). Use virutalenv, pyenv or conda env to do that.

```setup
conda create --name fc-env python=3.8.8
conda activate fc-env
```

Install dependencies:
```setup
git clone https://github.com/d-vet-ml4ed/fc-clustering
cd fc-clustering
pip install -r requirements.txt
```

The features used (BouroujeniEtAl, MarrasEtAl, LalleConati, and ChenCui) were extracted according to [Marras implementation](https://github.com/epfl-ml4ed/flipped-classroom) and placed in the data folder. 

To run the pipeline, define the feature groups in src/project_settings.py.

Then, run the gridsearch for the parameters of the first clustering step:

```python
python src/gridsearch/groups.py
```

Finally, run the clustering gridsearch for the second clustering step:

```python
python src/gridsearch/profiles.py
```


## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know.

Please feel free to file issues and pull requests on the repo and we will address them as we can.


## Citations
If you find this code useful in your work, please cite our paper:
```setup
@inproceedings{mejia2022,
  title={Identifying and Comparing Multi-Dimensional Student Profiles across Flipped Classrooms},
  author={Mejia-Domenzain, Paola and Marras, Mirko and Giang, Christian and Kaeser, Tanja} 
  booktitle={International Conference on Artificial Intelligence in Education},
  year={2022},
  organization={Springer}
}
```


## License
This code is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for details.

You should have received a copy of the GNU General Public License along with this source code. If not, go the following link: http://www.gnu.org/licenses/.


