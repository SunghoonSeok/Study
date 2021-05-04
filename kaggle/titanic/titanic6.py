import numpy as np 
import pandas as pd 

sub = pd.read_csv('c:/data/kaggle/titanic/sample_submission.csv')

sub1 = pd.read_csv('c:/data/kaggle/titanic/voting_submission3.csv') 
sub2 = pd.read_csv('c:/data/kaggle/titanic/submission.csv') 
sub3 = pd.read_csv('c:/data/kaggle/titanic/predict4.csv') 
sub4 = pd.read_csv('c:/data/kaggle/titanic/predict5.csv')

res = (sub1['Survived'] + sub2['Survived'] + sub3['Survived'] + sub4['Survived'])/4
sub.Survived = np.where(res > 0.5, 1, 0).astype(int)

sub.to_csv("c:/data/kaggle/titanic/submission3.csv", index = False)
sub['Survived'].mean()