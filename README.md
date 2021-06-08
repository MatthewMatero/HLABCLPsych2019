# HLAB CLPsych 2019

This repository stores the deep learning based code for [Suicide Risk Assessment with Multi-level
Dual-Context Language and BERT](https://www.aclweb.org/anthology/W19-3005.pdf), SBU-HLAB's 2019 CLPsych submission. All non-deep learning models used the [DLATK package](https://github.com/dlatk/dlatk) for quick iteration on logistic regression models. 

If you have any questions regarding the paper please contact mmatero [at] cs.stonybrook.edu (PhD student) or has [at] cs.stonybrook.edu (Lab director). 


# Details

The repo holds the PyTorch model defintiion of our attenion-based LSTM network, which scored on a F1 of .50 on Task A data. Alternatively, one could use this class file to instantiate a dual-context variant, used for task B, which uses Task A data (suicide-context) and task C data (non-suicide context) as described in the paper. 

# cite
```
@inproceedings{matero2019suicide,
  title={Suicide risk assessment with multi-level dual-context language and BERT},
  author={Matero, Matthew and Idnani, Akash and Son, Youngseo and Giorgi, Salvatore and Vu, Huy and Zamani, Mohammad and Limbachiya, Parth and Guntuku, Sharath Chandra and Schwartz, H Andrew},
  booktitle={Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology},
  pages={39--44},
  year={2019}
}
```
