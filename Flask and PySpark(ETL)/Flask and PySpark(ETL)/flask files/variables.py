import pandas as pd
import numpy as np

df1=pd.read_csv('Customer_and_bank details_p1.csv')
df2=pd.read_csv('Customer_campaign_details_p1.csv')
df5=pd.read_csv('Customer_social_economic_data_p1.csv')
agem=df1.age.mean()
agestd=df1.age.std()
previousm=df2.previous.mean()
previousstd=df2.previous.std()
campaignm=df2.campaign.mean()
campaignstd=df2.campaign.std()
empmean=df5['emp.var.rate'].mean()
empstd=df5['emp.var.rate'].std()
conspricemean=df5['cons.price.idx'].mean()
conspricestd=df5['cons.price.idx'].std()
euribormean=df5['euribor3m'].mean()
euriborstd=df5['euribor3m'].std()
consconfmean=df5['cons.conf.idx'].mean()
consconfstd=df5['cons.price.idx'].mean()
emplomean=df5['nr.employed'].mean()
emplostd=df5['nr.employed'].std()


labels=[{'admin.': 0,
  'blue-collar': 1,
  'entrepreneur': 2,
  'housemaid': 3,
  'management': 4,
  'retired': 5,
  'self-employed': 6,
  'services': 7,
  'student': 8,
  'technician': 9,
  'unemployed': 10},
 {'divorced': 0, 'married': 1, 'single': 2},
 {'high.school': 0,
  'illiterate': 1,
  'middle.school': 2,
  'professional.course': 3,
  'university.degree': 4},
 {'cellular': 0, 'telephone': 1},
 {'HIGH': 0, 'LOW': 1, 'MEDIUM': 2},
 {'no': 0, 'yes': 1},
 {'failure': 0, 'nonexistent': 1, 'success': 2},
 {'Central': 0, 'East': 1, 'South': 2, 'West': 3}]

job=labels[0]
marital=labels[1]
education=labels[2]
contact=labels[3]
duration=labels[4]
pdays=labels[5]
poutcome=labels[6]
region=labels[7]