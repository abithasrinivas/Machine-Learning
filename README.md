# Subgradient methods for the optimal soft margin hyperplane

Created as a part of EECS 545 (Machine Learning) course at University of Michigan 

Data : Download  the  file nuclear.mat.   The  variables x and y contain  training  data  for  a  binary classification problem.  
       The variables correspond to the total energy and tail energy of waveforms producedby a nuclear particle detector.  
       The classes correspond to neutrons and gamma rays. 
    
Implementation : 

<a href="https://www.codecogs.com/eqnedit.php?latex=J(w,b)&space;=\frac{1}{n}&space;\sum_{i=1}^{n}L(y_{i},w^{T}x_{i}&plus;b)&space;&plus;&space;\frac{\lambda}{2}\left&space;\|&space;w&space;\right&space;\|_{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(w,b)&space;=\frac{1}{n}&space;\sum_{i=1}^{n}L(y_{i},w^{T}x_{i}&plus;b)&space;&plus;&space;\frac{\lambda}{2}\left&space;\|&space;w&space;\right&space;\|_{2}" title="J(w,b) =\frac{1}{n} \sum_{i=1}^{n}L(y_{i},w^{T}x_{i}+b) + \frac{\lambda}{2}\left \| w \right \|_{2}" /></a>

L is hinge loss

We first implement subgradient method for minimizing J and apply it to the nuclear data and then implement stochiastic subgradient method, which is like the subgradient method, except that  the  step direction  is  a  subgradient  of a  random Ji,  not J.
