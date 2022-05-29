# PLDA
This repository is the official implementation of the PARM algorithm of the paper "Partial Label Learning with Discrimination Augmentation" and technical details of this algorithm can be found in the paper. 

## Requirements
- MATLAB, version 2014a and higher.

To start, create a directory of your choice and copy the code there.

Set the path in your MATLAB to add the directory you just created.
Then, run this command to enter the MATLAB environment:
```
matlab
```
## Demo
This repository provides a demo which shows the training and testing phase of the PLDA approach coupled with one of the state-of-the-art partial label learning (PLL) approach, PL-kNN. The coupled version for other PLL approaches can be implemented easily by replacing PL-kNN with the chosen  PLL algorithm.

To run demo.m, run this command in MATLAB command:

```
demo
```

## Citation
```
@inproceedings{NeurIPS20Wang,
    author = {Wang, Wei and Zhang, Min-Ling},
    title = {Partial label learning with discrimination augmentation},
    booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
    address = {Washington, DC},
    year = {2022},
    pages = {in press}
}
```
