# -*- coding: UTF-8 -*-
from sklearn.model_selection import train_test_split
def get_data(filename_train="zhihu_train.txt",filename_label="zhihu_label.csv"):
  file_train=open(filename_train)
  file_label=open(filename_label)
  data_x=[]
  data_y=[]
  dict_voca={}
  dict_label={}
  length=22
  title=0;
  for line in file_train:
    line=line.replace("\n","").split(" ")
    list_x=[]
    for split in line:
      if(split!="\r"):
        split=split.decode("utf-8")
        if(split not in dict_voca):
          dict_voca[split]=len(dict_voca)
        list_x.append(int(dict_voca[split]))
    i=0
    while(len(list_x)<length):
      list_x.append(list_x[i])
      i+=1
    data_x.append(list_x)
  for line in file_label:
    if(title==0):
      title=1
      continue
    line=line.replace("\n","").split(",")
    if(line[6] not in dict_label):
      dict_label[line[6]]=len(dict_label)
    #list_temp=[]
    data_y.append(int(dict_label[line[6]]))
    #train_y.append(list_temp)
  #print len(train_x)
  #print(len(dict_label))
  file_train.close()
  file_label.close()
  train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.1,random_state=15)
  return train_x,test_x,train_y,test_y,len(dict_voca)
def get_data_dynamic(filename_train="zhihu_train.txt",filename_label="zhihu_label.csv"):
  length=22
  file_train=open(filename_train)
  file_label=open(filename_label)
  data_x=[]
  data_y=[]
  dict_voca={}
  dict_label={}
  title=0;
  for line in file_train:
    line=line.replace("\n","").split(" ")
    list_x=[]
    for split in line:
      if(split!="\r"):
        split=split.decode("utf-8")
        if(split not in dict_voca):
          dict_voca[split]=len(dict_voca)
        list_x.append(int(dict_voca[split]))
    while(len(list_x)<length):
      list_x.append(0)
    data_x.append(list_x)
  for line in file_label:
    if(title==0):
      title=1
      continue
    line=line.replace("\n","").split(",")
    if(line[6] not in dict_label):
      dict_label[line[6]]=len(dict_label)
    data_y.append(int(dict_label[line[6]]))
  file_train.close()
  file_label.close()
  train_x,test_x,train_y,test_y=train_test_split(data_x,data_y,test_size=0.1,random_state=1)
  return train_x,test_x,train_y,test_y,len(dict_voca)
if __name__ == "__main__":
  get_data()
    
