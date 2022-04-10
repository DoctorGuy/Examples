import sys
import scipy
import numpy as py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns


df = pd.read_csv('C:/Users/ellio_000/Documents/IHME/2019projections.csv')
Train, Test = train_test_split(df, test_size =.2, random_state = 42)

## Questions I'll aim to answer: Who is a rising star? How accurate is the injury status
## for each position? How accurate is ESPN's projections in 2019?

## First let's create some simple plots

plt.bar(df['Pos'], df.iloc[:,0],  align='center', color='black', edgecolor='black')
plt.xlabel('Player position')
plt.ylabel('# of Players')
plt.title('Position distribution')
plt.show()
plt.clf()
plt.cla()


## Surprise it's even
## but what if I only want to see fantasy team 11's players?

df1 = df.loc[df['Team'] == 11]
height = range(1, 216)
plt.bar(df1['Pos'] , height,  align='center', color='blue', edgecolor='green')
plt.xlabel('Player position')
plt.ylabel('# of Players')
plt.title('My distribution')
plt.show()
plt.clf()
plt.cla()

## still pretty even and duh there are more flex players
## what about points by position?

plt.bar(df['Pos'] , df['Actual'],  align='center', color='blue', edgecolor='green')
plt.xlabel('Player position')
plt.ylabel('# of Players')
plt.title('Points by Position')
plt.show()
plt.clf()
plt.cla()

## love to see the negative points formt he bench thanks y'all
## okay we get it he can make a bar plot
## what if I wanted to look at just the my running backs performance over
## The season?
df2 = df1.loc[df1['Pos'] == 'RB']
Colors = {'Ezekiel Elliott':'red', 'Joe Mixon':'blue', 'Carlos Hyde':'green'}
x = df2['Week']
y = df2['Actual']
plt.scatter(x,y,c = df2['Player'].map(Colors))
plt.show()
plt.clf()
plt.cla()

##Why didn't you use seaborn? Well cause I wanted to show I can map colors to a dictionary
## but if you insist here it is, plus a regression line
sns.scatterplot(x,y , data = df2, hue = 'Player')
plt.show()
plt.clf()
plt.cla()

sns.lmplot('Week','Actual', data = df2, hue = 'Player', fit_reg = True)
plt.show()
plt.clf()
plt.cla()

## And this led me to draft zeke two more times... a mistake I won't make a third
# lets try and answer some question huh? So who was a rising star? To answer
# this I'll take the difference from projected and actual and whoever had the highest
## difference will be our rising star

df['star'] = df['Actual'] - df['Proj']
x1 = df['Week']
y1 = df['star']
sns.scatterplot(x1, y1, data = df, hue = 'Pos')
plt.show()
plt.clf()
plt.cla()
##omg a receiver got 40 more than projected? Painful week for that fantasy owner
## playing against them...

df['star'] = df['Actual'] - df['Proj']
sumarray = df.groupby(['Player'], as_index =False)['star'].sum()

sumarray = sumarray.nlargest(n=5, columns = ['star'])

plt.bar(sumarray['Player'] , sumarray['star'],  align='center', color='blue', edgecolor='green')
plt.xlabel('Player')
plt.ylabel('Over total projected')
plt.title('Top 5 stars')
plt.show()
plt.clf()
plt.cla()


## The ever elusive Lamer Jackson crushed it and CMC, oh god CMC how you have hurt
## me since then
## anywho let's look at some injuries
