# -- coding: utf-8 --
"""
Created on Sun Apr 25 23:49:30 2021

@author: tejasv
"""
import tkinter
import csv
import math
import random
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

extra_characters = [
    '.',
    '?',
    '-',
    ',',
    '/'
    ]

def data_cleanse(list1):
    res = []
    for ele in list1:
        for l in extra_characters:
            ele = ele.replace(l,'')
        res.append(ele)
    return res

degree_data = pd.read_csv("degrees-that-pay-back.csv")
degrees = degree_data["Undergraduate Major"]
degrees = [each_word.lower() for each_word in degrees]
degrees = data_cleanse(degrees)

college_data = pd.read_csv("salaries-by-college-type.csv")
colleges = college_data["School Name"]
colleges = [each_word.lower() for each_word in colleges]
colleges = data_cleanse(colleges)
college_types = college_data["School Type"].unique()
college_types = [each_word.lower() for each_word in college_types]
college_types = data_cleanse(college_types)

region_data = pd.read_csv("salaries-by-region.csv")
regions = region_data["Region"].unique()
regions = [each_word.lower() for each_word in regions]
regions = data_cleanse(regions)

event_data = pd.read_csv("Event.csv")
events = event_data["Events"].unique()
events = [each_word.lower() for each_word in events]
events = data_cleanse(events)

question_words = {
    "which",
    "what",
    "how",
    "where",
    "who",
    "whose"
    }

question_first_words = {
    "can",
    "is",
    "whether",
    "if",
    "shall",
    "do",
    "name",
    "list",
    "give",
    "will"
    }

excess_words = {
    "is",
    "the",
    "a",
    "of",
    "in",
    "are",
    "you",
    "me",
    "at"
    }


quantitative_keywords = [
    ("top", #Top
    "most",
    "best",
    "highest"),
    ("bottom",  #Bottom
    "least",
    "worst",
    "lowest"),
    ("middle", #Average
    "mean",
    "median",
    "good",
    "average")
    ]

subject_keywords = [
    ("money",   #Money
    "salary",
    "salaries"),
    ("job",   #Job
     "jobs",
    "major",
    "majors",
    "subject",
    "subjects",
    "career",
    "careers"),
    ("college",  #College
     "colleges"
     "school",
     "schools",
     "university",
     "universities"),
    ("region",
     "regions")    #Region
    ]

prediction_keywords = [
    'chance',
    'chances',
    'get in',
    'get a seat',
    'get'
    ]

fair_keywords = [
    'event',
    'fair',
    'competition',
    'challenge',
    'party'
    ]

def chatbot_startup():
    msg = "Hello there! How may I help you?"
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "Bot: " + msg + '\n\n')
    #ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)
          
def common_elements(list1,list2):
    common_keyword = []
    common_ip = []
    # Iterate in the 1st list 
    for m in list1: 
  
        # Iterate in the 2nd list 
        for n in list2: 
            
            if type(n) is tuple: # Predefined keywords
                if m in n:
                    common_keyword.append(n[0])
                    common_ip.append(m)
                    
            else: # Dataset keywords
                o = n.split()
                for ele in o:
                    # if there is a match
                    if m == ele: 
                        common_keyword.append(n)
                        common_ip.append(m)
    
    return (common_ip,common_keyword)

def best_match(list1,string):
    best = 0
    highest = 0
    val = []
    for ind,ele in enumerate(list1):
        #print(ele)
        if ele in string:
            #print("direct match")
            return ele
        
        valup = 0
        valdown = 0
        substring = ele
        for x in range(len(ele),-1,-1):
            substring = substring.replace(substring,substring[:-1])
            if substring in string:
                #print("'",substring,"'")
                valdown = len(substring)
                break
        substring = ele
        for x in range(len(ele)):
            substring = substring.replace(substring,substring[1:])
            if substring in string:
                #print("'",substring,"'")
                valup = len(substring)
                break
        if valup == 0 and valdown == 0:
            words = ele.split()
            for l in words:
                if l in string:
                    val.append(len(l))
                    break    
        else:
            val.append(max(valup,valdown))
        #print("valdown is ",valdown," valup is ",valup," val is ",val)
    #print("final val is ",val)
    if len(val)!=0:
        highest = max(val)
    else:
        highest = 0
    #print("highest is ",highest)
    best = [i for i, x in enumerate(val) if x == highest]
    #print(" best is ",best)
    out = []
    for o in best:
        out.append(list1[o])
    return out

def handle_overlap_college_career(ipw,ip,ind,ip_keywords):
    if(ind==0 or ind==5 or ind==1):
        return (ipw,ip)
    # if():#Before colleges
    #     overlap = subject_keywords
    if(ind==2):#Before regions
        overlap = colleges
    if(ind==3):#Before colleges types
        overlap = regions
    if(ind==4):#Before degrees
        overlap = college_types
    ipw = list(set(ipw).difference(set(overlap)))
    for x in ip:
        ip = ip.replace(x,'')
    return (ipw,ip)
        
def process_college_career_keywords(ip_keywords):
    if len(ip_keywords[5])!=0 :#If degree exists
        ip_keywords[4] = [] #College type should not exist
    if len(ip_keywords[2])!=0 :#If college exists
        if len(ip_keywords[2])>1 :#If more than one college exists
            for x in regions:
                if x in ip_keywords[2][0]:
                    ip_keywords[2] = [] #colleges are removed and region is added
                    ip_keywords[3] = [x]
                    break
    return ip_keywords

def get_college_career_keywords(ip_words,ip):
    
    college_career_keywords = [quantitative_keywords,subject_keywords,colleges,regions,college_types,degrees]
    ip_keywords = []
    common_ip = []
    common_keyword = []
    common_dataset = []
    
    ipk = list(set(ip_words).difference(excess_words))
    ipk = list(set(ipk).difference(question_words))
    ipk = list(set(ipk).difference(question_first_words))
    ipc = ipk
    
    for ind,ele in enumerate(college_career_keywords):
        
        (common_ip,common_keyword) = common_elements(ipk, ele)
        if(ind>1): #For dataset
            #print(ind)
            common_keyword = list(set(common_keyword))
            common_dataset = [ele for ele in common_keyword if ele in ip]
            #print("common_dataset is",common_dataset)
            #print("common_keyword is",common_keyword)
            common_keyword = best_match(common_keyword,ip)
            if type(common_keyword) == str:
                common_keyword = [common_keyword]
            #print("common_keyword is",common_keyword)
            #print("common_keyword type is",type(common_keyword))
            #print("common_dataset type is",type(common_dataset))
            if len(common_dataset)!=0 and len(common_dataset)<len(common_keyword) :
                common_dataset = best_match(common_dataset,ip)
                if type(common_keyword) == str:
                    common_keyword = [common_keyword]
                print("common_dataset is",common_dataset)
                common_keyword = set(common_keyword).intersection(common_dataset)
            #print("common_keyword is",common_keyword)
        ip_keywords.append(common_keyword)
        ipk = list(set(ipk).difference(set(common_ip)))
        #(ipk,ip) = handle_overlap_college_career(ipk,ip,ind,ip_keywords)
        
    ip_keywords = process_college_career_keywords(ip_keywords)
    return tuple(ip_keywords)

def handle_quant_cond(row_r,quant_cond):
    if quant_cond == 'top':
        max_value = row_r['Starting Median Salary'].max()
        row = row_r.loc[row_r['Starting Median Salary'] == max_value]
        op = str(row["School Name"].tolist()[0]) + " has starting salary of " + max_value + ".\n"
    elif quant_cond == 'bottom':
        min_value = row_r['Starting Median Salary'].min()
        row = row_r.loc[row_r['Starting Median Salary'] == min_value]
        op = str(row["School Name"].tolist()[0]) + " has starting salary of " + min_value + ".\n"
    elif quant_cond == 'middle':
        col = row_r['Starting Median Salary'].replace({'\$': '', ',': ''}, regex=True).astype(float)
        #print(col)
        mean_value = col.mean()
        #print(mean_value)
        result_index = col.sub(mean_value).abs().idxmin()
        #print(result_index)
        row = row_r.loc[result_index]
        #print(row)
        op = row["School Name"] + " has starting salary of " + row["Starting Median Salary"] + ".\n"
    return op

# l = [[quantitative_keywords],[subject_keywords],[college_name],[region],[college_type],[degree_major]]

def get_college_career_data(l):
    degree_data_lc = degree_data.applymap(lambda s:s.lower() if type(s) == str else s)
    college_data_lc = college_data.applymap(lambda s:s.lower() if type(s) == str else s)
    region_data_lc = region_data.applymap(lambda s:s.lower() if type(s) == str else s)
    if len(l[2]) != 0: #College or University Name
        row = college_data_lc.loc[college_data_lc['School Name'] == l[2][0]]
        op = str(row["School Name"].tolist()[0]) + " has starting salary of " + str(row["Starting Median Salary"].tolist()[0] + ".\n")
    elif len(l[5]) != 0: #Degree Major
        row = degree_data_lc.loc[degree_data_lc['Undergraduate Major'] == l[5][0]]
        op = str(row["Undergraduate Major"].tolist()[0]) + " has starting salary of " + str(row["Starting Median Salary"].tolist()[0] + ".\n")
    elif len(l[3]) != 0: #Regions
        if len(l[0]) != 0:
            row_r = region_data_lc.loc[region_data_lc['Region'] == l[3][0]]
            op = handle_quant_cond(row_r,l[0][0])
    elif len(l[4]) != 0: #College Type (State, ivy league, engineering, liberal arts, ...)
        if len(l[0]) !=0:
            row_r = college_data_lc.loc[college_data_lc['School Type'] == l[4][0]]
            op = handle_quant_cond(row_r,l[0][0])
    elif len(l[0]) !=0: #Quantitative (top, good, low rated)
        row_r = college_data_lc
        op = handle_quant_cond(row_r,l[0][0])
    
    return op

def process_college_event_keywords(ip_keywords):
    if len(ip_keywords[1])!=0 :#If college exists
        ip_keywords[0] = []
        if len(ip_keywords[1])>1 :#If more than one college exists
            for x in regions:
                if x in ip_keywords[1][0]:
                    ip_keywords[1] = [] #colleges are removed and region is added
                    ip_keywords[2] = [x]
                    break
    return ip_keywords

def get_college_event_keywords(ip_words,ip):
    
    college_event_keywords = [subject_keywords,colleges,regions,events]
    ip_keywords = []
    common_ip = []
    common_keyword = []
    common_dataset = []
    
    ipk = list(set(ip_words).difference(excess_words))
    ipk = list(set(ipk).difference(question_words))
    ipk = list(set(ipk).difference(question_first_words))
    ipc = ipk
    
    for ind,ele in enumerate(college_event_keywords):
        
        (common_ip,common_keyword) = common_elements(ipk, ele)
        if(ind>1): #For dataset
            #print(ind)
            common_keyword = list(set(common_keyword))
            common_dataset = [ele for ele in common_keyword if ele in ip]
            #print("common_dataset is",common_dataset)
            #print("common_keyword is",common_keyword)
            common_keyword = best_match(common_keyword,ip)
            if type(common_keyword) == str:
                common_keyword = [common_keyword]
            #print("common_keyword is",common_keyword)
            #print("common_keyword type is",type(common_keyword))
            #print("common_dataset type is",type(common_dataset))
            if len(common_dataset)!=0 and len(common_dataset)<len(common_keyword) :
                common_dataset = best_match(common_dataset,ip)
                if type(common_keyword) == str:
                    common_keyword = [common_keyword]
                print("common_dataset is",common_dataset)
                common_keyword = set(common_keyword).intersection(common_dataset)
            #print("common_keyword is",common_keyword)
        ip_keywords.append(common_keyword)
        ipk = list(set(ipk).difference(set(common_ip)))
        #(ipk,ip) = handle_overlap_college_career(ipk,ip,ind,ip_keywords)
        
    ip_keywords = process_college_event_keywords(ip_keywords)
    return tuple(ip_keywords)

def get_event_data(ip):
    
    event_data_lc = event_data.applymap(lambda s:s.lower() if type(s) == str else s)
    op = ''
    #college_event_keywords = ['subject_keywords','colleges','regions','events']
    
    if len(ip[1]) != 0:
        temp = event_data[event_data_lc['School Name'] == ip[1][0]]
        if str(temp['No Of Covid Cases'].tolist()[0]) == 'High':
            op = ip[1][0] + " has an event of " + str(temp['Events'].tolist()[0]) + "." + "Since no of covid cases are high in that region, please avoid attending the event. \n"
        elif str(temp['No Of Covid Cases'].tolist()[0]) == 'Medium':
            op = ip[1][0] + " has an event of " + str(temp['Events'].tolist()[0]) + "." + " Since no of covid cases are medium in that region, take precautionary measures while going out and always wear mask. \n "
        else:
             op = ip[1][0] + " has an event of " + str(temp['Events'].tolist()[0]) + "." + " Since no of covid cases are less in that region, always wear a mask while going out. \n "
       
            
            
            
        #op = ip[1][0] + " has an event of " + str(temp['Events'].tolist()[0]) + "."
        
    elif len(ip[2]) != 0:
        temp = event_data_lc[event_data_lc['Region'] == ip[2][0]]
        if temp.shape[0] == 1:
            op = ip[2][0] + " region has an event of " + str(temp['Events'].tolist()[0]) + "."
        else:
           xyz = temp.sample(3)
           
           if str(temp['No Of Covid Cases'].tolist()[0]) == 'High':
                op = ip[2][0] + " region has " + str(temp.shape[0]) + " events including " + str(xyz['Events'].tolist()[0]) + "," + str(xyz['Events'].tolist()[1]) + "," + str(xyz['Events'].tolist()[2]) + "." + "Since no of covid cases are high in that region, please avoid attending the event. \n"
            
           elif str(temp['No Of Covid Cases'].tolist()[0]) == 'Medium':
                op = ip[2][0] + " region has " + str(temp.shape[0]) + " events including " + str(xyz['Events'].tolist()[0]) + "," + str(xyz['Events'].tolist()[1]) + "," + str(xyz['Events'].tolist()[2]) +  "." + " Since no of covid cases are medium in that region, take precautionary measures while going out and always wear mask. \n "
           else:
                op = ip[2][0] + " region has " + str(temp.shape[0]) + " events including " + str(xyz['Events'].tolist()[0]) + "," + str(xyz['Events'].tolist()[1]) + "," + str(xyz['Events'].tolist()[2]) +  "." + " Since no of covid cases are less in that region, always wear a mask while going out. \n "
 
           #op = ip[2][0] + " region has " + str(temp.shape[0]) + " events. And some events are " + str(xyz['Events'].tolist()[0]) + "," + str(xyz['Events'].tolist()[1]) + "," + str(xyz['Events'].tolist()[2]) + "." 
           
              
    elif len(ip[3]) != 0:
        temp = event_data_lc[event_data_lc['Events'] == ip[3][0]]
        if temp.shape[0] == 1:
            op = ip[3][0] + " will be held at " + str(temp['School Name'].tolist()[0]) + "."
        else:
            xyz = temp.sample(3)
            op = ip[3][0] + " event will be held at " + str( temp.shape[0]) + " colleges including " + str(xyz['School Name'].tolist()[0]) + ", " + str(xyz['School Name'].tolist()[1]) + " and " + str(xyz['School Name'].tolist()[2]) + ". Always wear mask while you go out. \n" 
        
                          
                          
    return op

def process_user_ip(user_ip):
    user_ip = user_ip.strip()
    if len(user_ip)==0:
        print("Please enter a valid response!")
        return False
    else:
        return True

def convert_nested_tuple_to_str(tuple1):
    s = "["
    for ele in tuple1:
        print(ele)
        if len(ele)==0:
            s=s+"[],"
        else:
            s=s+str(ele)+","
    s=s[:-1]
    s=s+"]"
    return s

def list_storage_conversion(ip_list):
    op_list = []
    for ele in ip_list:
        s = ""
        if type(ele) == list and len(ele) == 1:
            op_list.append(ele[0])
        elif type(ele) == list and len(ele) > 1:
            for e in ele:
                s = s + str(e) + ";"
            s = s[:-1]
            op_list.append(s)
        elif type(ele) == list and len(ele) == 0:
            op_list.append("")
        else:
            op_list.append(ele)
            
    return op_list

def store_response(ip_list):
    if ip_list[0][0] == 'event':
        ip_list.insert(1,[])
        ip_list.insert(5,[])
        ip_list.insert(6,[])
    ip_list = list_storage_conversion(ip_list)
    with open('Responses.csv', 'a', newline='') as csv_file:
        response_writer = csv.writer(csv_file)
        response_writer.writerow(ip_list)
        
def get_fact(*word):
    op = ''
    order = [[college_data,'School Name'],[region_data,'Region'],[college_data,'School Type'],[degree_data,'Undergraduate Major'],[event_data,'Events']]
    if len(word) == 0:
        ind = random.randint(0,len(order)-1)
        #row = order[ind][0].loc[order[ind][0].applymap(lambda s:s.lower() if type(s) == str else s)[order[ind][1]]==word]
        row = order[ind][0].sample()
        row = row.dropna()
        while row.empty:
            ind = random.randint(0,len(order)-1)
            row = order[ind][0].sample()
            row = row.dropna()
        # print("ind",ind)
        # print("row",row)
        s = row.columns.tolist()
        # print("s",s)
        # print("s[0]",s[0])
        st = s.copy()
        if ind != 4 :
            st.remove(order[ind][1])
        # print("st",st)
            l = st[random.randint(0,len(st)-1)]
        else:
            l = order[ind][1]
        while l=='School Name':
            l = st[random.randint(0,len(st)-1)]
        # print("l",l)
        # print("row[s[0]]",row[s[0]])
        # print("row[s[0]].tolist()[0]",row[s[0]].tolist()[0])
        # print("row[l].tolist()[0]",row[l].tolist()[0])
        op = op + str(row[s[0]].tolist()[0]) + " students have " + l + " of " + str(row[l].tolist()[0])
        
    else:
        for ind,ele in enumerate(order):
            # print("ind ",ind)
            if order[ind][0].applymap(lambda s:s.lower() if type(s) == str else s)[order[ind][1]].isin([word[0]]).any().any():
                row = order[ind][0].loc[order[ind][0].applymap(lambda s:s.lower() if type(s) == str else s)[order[ind][1]]==word[0]]
                row = row.dropna()
                if row.shape[0] > 1:
                    row = row.sample()
                s = row.columns.tolist()
                st = s.copy()
                if ind != 4 :
                    st.remove(order[ind][1])
                # print("s",s)
                    l = st[random.randint(0,len(st)-1)]
                else:
                    l = order[ind][1]
                # print("st",st)
                # print("row",row)
                while l=='School Name':
                    l = st[random.randint(0,len(st)-1)]
                # print("l",l)
                # print('row[s[0]].tolist()[0]',row[s[0]].tolist()[0])
                # print("row[l].tolist()[0]",row[l].tolist()[0])
                op = ". Did you know that " + str(row[s[0]].tolist()[0]) + " students have " + l + " of " + str(row[l].tolist()[0])
                break
        
        if word[0] == 'top':
            op = " colleges. Aim for the sky! "
        elif word[0] == 'bottom':
            op = " colleges. Hmm... "
        elif word[0] == 'middle':
            op = " colleges. Be more ambitious! "
        elif word[0] == 'money':
            op = ". Career is more than money! "
        elif word[0] == 'prediction':
            op = ". Don't be so anxious! It will be alright!"
    
    return op
  

def random_reply(*response):
    r = random.randint(1,4)
    op = ""
    #Funfact
    #Quiz
    #Query
    if len(response) == 0:
        if r == 1 or r == 2 or r == 3:
            op = "Did you know that " + get_fact()
        elif r == 4:
            op = "Do you have any more queries?"
    else:
        op = "You seem to be interested in " + response[0]
        if r == 1 or r == 2 or r == 3:
            op = op + get_fact(response[0])
        elif r == 4:
            op = "Do you have any more queries?"
    
    return op
    

def converse():
    responses_data = pd.read_csv("Responses.csv")
    if len(responses_data.index)>2:
        responses_data = responses_data.tail(3)
        # print(responses_data)
        
        college = responses_data[responses_data.duplicated(["Colleges"])]["Colleges"].tolist()
        college = [x for x in college if pd.isnull(x) == False]
        # print(college)
        
        college_common = responses_data["Colleges"].tolist()
        college_common = [x for x in college_common if pd.isnull(x) == False]
        if len(college_common)>=2:
            if college_data.loc[college_data['School Name'] == college_common[-1]]['School Type'] == college_data.loc[college_data['School Name'] == college_common[-2]]['School Type']:
                college_common = college_data.loc[college_data['School Name'] == college_common[-1]]['School Type']
            elif region_data.loc[region_data['School Name'] == college_common[-1]]['Region'] == region_data.loc[region_data['School Name'] == college_common[-2]]['Region']:
                college_common = region_data.loc[college_data['School Name'] == region_data[-1]]['Region']
        # print(college_common)
        
        query_Type = responses_data[responses_data.duplicated(["Query Type"])]["Query Type"].tolist()
        query_Type = [x for x in query_Type if pd.isnull(x) == False]
        # print(query_Type)
        
        quantitative_keyword = responses_data[responses_data.duplicated(["Quantitative keywords"])]["Quantitative keywords"].tolist()
        quantitative_keyword = [x for x in quantitative_keyword if pd.isnull(x) == False]
        # print(quantitative_keyword)
        
        subject_Keyword = responses_data[responses_data.duplicated(["Subject Keywords"])]["Subject Keywords"].tolist()
        subject_Keyword = [x for x in subject_Keyword if pd.isnull(x) == False]
        # print(subject_Keyword)
        
        region = responses_data[responses_data.duplicated(["Regions"])]["Regions"].tolist()
        region = [x for x in region if pd.isnull(x) == False]
        # print(region)

        college_type = responses_data[responses_data.duplicated(["college Types"])]["college Types"].tolist()
        college_type = [x for x in college_type if pd.isnull(x) == False]
        # print(college_type)

        degree = responses_data[responses_data.duplicated(["Degrees"])]["Degrees"].tolist()
        degree = [x for x in degree if pd.isnull(x) == False]
        # print(degree)

        event = responses_data[responses_data.duplicated(["Events"])]["Events"].tolist()
        event = [x for x in event if pd.isnull(x) == False]
        # print(event)
        
        priority = [college,college_common,region,college_type,degree,event,quantitative_keyword,query_Type,subject_Keyword]
        priority_response = ""
        
        if any(priority):
            for ele in priority:
                if any(ele):
                    priority_response = ele[0]
                    break
        
        if any(priority_response):
            return random_reply(priority_response)
        else:
            return random_reply()
        
    else :
        return random_reply()       

reply = 0
prediction_data = []
# GRE, TOEFL, University, Sop, LOR, CGPA, Research

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def admission_predict(value):
    import numpy as np
    import pandas as pd
    from sklearn import svm
    import seaborn as sns
    import matplotlib.pyplot as plt
    dataset=pd.read_csv("Admission_Predict_Ver1.1.csv")
    dataset.head()
#Omitting the first column
    updated_dataset=dataset.iloc[:,1:9]
    updated_dataset.head()
    updated_dataset.describe()
#Checking for NA values
    updated_dataset.isna().sum()
    updated_dataset.corr(method="pearson")
    X=updated_dataset.iloc[:,:7]
    y=updated_dataset["Chance of Admit "]
    model = svm.SVR(gamma='scale', C=30, epsilon=0.0002,tol=0.001)
    model.fit(X,y)
    inputList = [ item for elem in value for item in elem]
    val = dict(zip(updated_dataset.columns[:-1], inputList))
    val = pd.Series(val)
    val = np.array(val).reshape(1, -1)
    y_pred=model.predict(val)
    percent=y_pred*100
    output = "Chances of your admission is %.2f" % round(percent.item(0),2)
    #plt.subplots(figsize=(5,3))
    #sns.barplot(x="University Rating",y="Chance of Admit ",data=updated_dataset)
    global plot_mode,plot_data
    plot_mode = 1
    plot_data = updated_dataset
    return output

def awaiting_reply(ip):
    op = ''
    if isfloat(ip):
        #print('here')
        ip = float(ip)
        #print("ip is",ip)
        global reply
        #print("reply",reply)
        global prediction_data
        if reply == 1 : #GRE
            #print("HERE 2")
            if ip >= 260 and ip <= 340:
                #print('here 3')
                prediction_data.append([ip])
                op = "Please enter your TOEFL score."
        elif reply == 2 : #TOEFL
            if ip >= 0 and ip <= 120:
                prediction_data.append([ip])
                op = "Please enter your desired University Rating in range of 0-5."
        elif reply == 3 : #University Rating
            if ip >= 0 and ip <= 5:
                prediction_data.append([ip])
                op = "Please enter your SOP Rating in range of 0-5."
        elif reply == 4 : #SOP
            if ip >= 0 and ip <= 5:
                prediction_data.append([ip])
                op = "Please enter your LOR Rating in range of 0-5."
        elif reply == 5 : #LOR
            if ip >= 0 and ip <= 5:
                prediction_data.append([ip])
                op = "Please enter your CGPA."
        elif reply == 6 : #CGPA
            if ip >= 0 and ip <= 10:
                prediction_data.append([ip])
                op = "Please enter whether you have done any research.[Enter 1 for Yes and 0 for No]"
        elif reply == 7 : #Research
            if ip >= 0 and ip <= 1:
                prediction_data.append([ip])
                op = admission_predict(prediction_data)
                prediction_data = []
        
        if op == '':
            op = "Please enter a valid score!"
            
        if op != "Please enter a valid score!":
            if reply == 7:
                reply = 0
            else:
                reply = reply + 1
    else:
            op = "Please enter a valid score!"
    
    return op

def check_for_question_and_handle(user_ip):
    user_ip = user_ip.strip()
    question_mark = False
    if '?' in user_ip:
        question_mark = True
        user_ip = user_ip.replace('?','')
        
    input_words = user_ip.split()
    input_words = [each_word.lower() for each_word in input_words]
    
    if question_words.intersection(set(input_words)) or input_words[0] in question_first_words or question_mark:
        
        #question_keywords = question_words.intersection(input_words) or input_words[0]
        
        if any(word in user_ip for word in prediction_keywords):
            global reply
            reply = 1
            out = "Please enter your GRE score."
            return (True,["prediction"],out)
        
        elif any(word in user_ip for word in fair_keywords):
            output = get_college_event_keywords(input_words,user_ip)
            output = list(output)
            out = get_event_data(output)
            output.insert(0,["event"])
            return (True,output,out)
        
        else:
            output = get_college_career_keywords(input_words,user_ip)
            output = list(output)
            if not any(output):
                return (False,[],'')
            out = get_college_career_data(output)
            output.insert(0,["college-career"])
            return (True,output,out)
    else:
        return (False,[],'')         

def chatbot_handle(user_ip):
    user_ip = user_ip.strip()
    bot_input = chatbot.get_response(user_ip)
    return str(bot_input)
            
def chatbot_response(user_ip):
    ip = user_ip
    op = ""
    op_list = []
    global reply
    if process_user_ip(ip):
        if reply == 0:
            (val,op_list,op_str) = check_for_question_and_handle(ip)
            if val:
                store_response(op_list)
                if len(op_str) == 0:
                    op = str(op_list) + ". "
                else:
                    op = op_str
                if reply == 0:
                    op = op + converse()
            else:
                op = chatbot_handle(ip)
        else:
            op = awaiting_reply(ip)
    return op

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        #ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = str(chatbot_response(msg))
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

out_plot = None
fig = None
plot_data = None
plot_mode = 0
def plot():
    import seaborn as sns
    global df1
    global out_plot,fig
    if not out_plot:
        # fig = plt.Figure()
        # ax1 = fig.add_subplot(111)
        
        
        # figure = Figure(figsize=(6, 6))
        # ax = figure.subplots()
        # sns.heatmap(matrix, square=True, cbar=False, ax=ax)
        # df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
        # df1.plot(kind='bar', legend=True, ax=ax1)
        # ax1.set_title('Country Vs. GDP Per Capita')
        if plot_mode == 1:
            fig, ax1 = plt.subplots(figsize=(5, 3))
            out_plot = FigureCanvasTkAgg(fig, base)
            out_plot.get_tk_widget().place(x=401,y=6)
            sns.barplot(x="University Rating",y="Chance of Admit ",data=plot_data, ax=ax1)
    else:
        clear_plot()
def clear_plot():
    global out_plot
    if out_plot:
        out_plot.get_tk_widget().destroy()
    out_plot = None   

with open('Responses.csv', 'w', newline='') as csv_file:
    fieldnames = ['Query Type','Quantitative keywords','Subject Keywords','Colleges','Regions','college Types','Degrees','Events']
    response_writer = csv.writer(csv_file)
    response_writer.writerow(fieldnames)

            
chatbot = ChatBot(
    "Terminal",
    storage_adapter="chatterbot.storage.SQLStorageAdapter",
    logic_adapters=[
        #"chatterbot.logic.MathematicalEvaluation",
        #"chatterbot.logic.TimeLogicAdapter",
        "chatterbot.logic.BestMatch"
    ],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
    database="../database.db"
)

trainer = ChatterBotCorpusTrainer(chatbot)

trainer.train('chatterbot.corpus.english')

""" Start of Main Code """

#chatbot_startup()
base = ThemedTk(theme="breeze")
s = ttk.Style()
#s.configure('my.TButton')
base.title("Tech it Out!")
base.geometry("850x500")
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50")
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = ttk.Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = ttk.Button(base, text="Send",style='my.TButton',
                    command= send )
PlotButton = ttk.Button(base, text="Plot",style='my.TButton',
                    command= plot )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90, width = 115)
PlotButton.place(x=600, y=401, height=90, width = 90)
base.after(1000, chatbot_startup())
base.mainloop()
