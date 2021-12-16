# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:41:42 2021

@author: Chai Wah Wu
"""

import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import re
import requests
import seaborn as sns
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sympy import factor, sympify, sin, cos, tan, sqrt, S, exp, log, Add, Mul, Pow, simplify, Number, Symbol, Float, expand
from sympy import ceiling, floor, Max, Min, Abs
import math
from sympy.abc import n
import urllib.parse
import plotly.express as px
import plotly.graph_objects as go

nameregex = re.compile('<tt>%N (.*)\n</tt>')
searchregex = re.compile('<a href="\/A(\d{6})">A(\d{6})<\/a>')

square = make_function(function=lambda x:x**2,name='square',arity=1)
cube = make_function(function=lambda x:x**3,name='cube',arity=1)
power = make_function(function = lambda x,y:x**y, name='pow',arity=2)
expf = make_function(function = lambda x: np.e**x, name = 'exp', arity=1)
ceilf = make_function(function = lambda x: np.ceil(x), name = 'ceil', arity=1)
floorf = make_function(function = lambda x: np.floor(x), name = 'floor', arity=1)

sympy_funcs = {
    'sub': lambda x,y: Add(x,-y),
    'div': lambda x,y: Mul(x,Pow(y,-1)),
    'mul': Mul,
    'add': Add,
    'neg': lambda x: -x,
    'pow': Pow,
    'sin': sin,
    'cos': cos,
    'tan': tan,
    'inv': lambda x: Pow(x,-1),
    'sqrt': sqrt,
    'square': lambda x: Pow(x,2),
    'cube': lambda x: Pow(x,3),
    'exp': exp,
    'log': log,
    'ceil': ceiling,
    'floor': floor,
    'max': Max,
    'min': Min,
    'abs': Abs
}


function_set_dict = {
    'x-y': 'sub',
    'x/y': 'div',
    'x*y': 'mul',
    'x+y': 'add',
    '-x' : 'neg',
    'sin(x)': 'sin',
    'cos(x)': 'cos',
    'tan(x)': 'tan',
    '1/x':  'inv',
    'sqrt(x)': 'sqrt',
    'x^2': square,
    'x^3': cube,
    'ceil': ceilf,
    'floor': floorf,
    'max': 'max',
    'min': 'min',
    'abs': 'abs'}

function_set_labels = list(function_set_dict.keys())

maxI = 10**308
num_digits = 4

def round_expr(expr, num_digits): return expr.xreplace({n.evalf() : Float(n, num_digits) for n in expr.atoms(Number) if n.is_Float})

@st.cache
def readfromOEIS(s): #read bfile and sequence information from OEIS
    OEISstruct = {}
    indexlist, datalist = [],[]
    OEISurl = 'https://oeis.org/A'+s+'/b'+s+'.txt'
    try:
        data = requests.get(OEISurl).text.split('\n')
    except:
        indexlist, datalist = [],[]
    else:
        for line in data:
            if not line.startswith('#'):
                try:
                    a,b = map(int,line.strip().split()) 
                except:
                    pass
                else:
                    indexlist.append(a)
                    datalist.append(b)
    OEISstruct['bfile'] = pd.DataFrame(index=indexlist,data=datalist,columns=['terms'])
    OEISinternalURL = 'https://oeis.org/A'+s+'/internal'
    text = requests.get(OEISinternalURL).text
    nametext = nameregex.search(text).groups()
    if len(nametext) > 0:
        OEISstruct['name']=nametext[0]
    return OEISstruct

class OEISsequence(object):
    def __init__(self, n):
        self.n = n
        self.s = str(n).zfill(6)
        self.OEISstruct = readfromOEIS(self.s)
        self.nterms = len(self.OEISstruct['bfile'])
    def get_data(self): 
        return (np.array(self.OEISstruct['bfile'].index), np.array(self.OEISstruct['bfile']['terms']))
    def get_data_trimmed(self):
        a, b = self.get_data()
        indexlist, data = [],[]
        for i in range(len(a)):
            if abs(b[i]) <= maxI:
                indexlist.append(a[i])
                data.append(b[i])
        return (np.array(indexlist), np.array(data))
    def plot_data(self):    
        if 'bfile' in self.OEISstruct:
            indexlist, data = self.get_data_trimmed()
            ylabel = 'A'+self.s+'(n)'
            df = pd.DataFrame(np.array([indexlist,data]).T)
            df.columns = ['n', ylabel]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=indexlist,y=data,mode='markers',name='terms'))
            fig.update_layout(xaxis_title='n',yaxis_title=ylabel)
            logfig = go.Figure()
            logfig.add_trace(go.Scatter(x=indexlist,y=data,mode='markers',name='terms'))
            logfig.update_layout(yaxis_type="log",xaxis_title='n',yaxis_title=ylabel)
            return fig, logfig

st.title('On-Line Encyclopedia of Integer Sequences ([OEIS](https://oeis.org/)) explorer')
st.write("A tool for exploring integer sequences in the [OEIS](https://oeis.org/)")

seqnum1 = 2378
seqnum2 = 345
st.header("First sequence")

search_terms = st.text_input('Search terms','A'+str(seqnum1),key='search1')
rx = searchregex.search(requests.get('https://oeis.org/search?q='+urllib.parse.quote_plus(search_terms)+'&language=english&go=Search').text)
if hasattr(rx,'groups') and len(rx.groups()) > 0:
    a, b = rx.groups()
    if a == b:
        seqnum1 = int(a)

sequence1 = OEISsequence(seqnum1)
if 'name' not in sequence1.OEISstruct or not hasattr(rx,'groups') or len(rx.groups()) == 0:
    st.write("Sequence not found. Please try again.")
else:
    st.write("OEIS sequence ",seqnum1,": ", sequence1.OEISstruct['name'])   
    fig1, logfig1 = sequence1.plot_data()
    indexlist, data = sequence1.get_data_trimmed()
    logindexlist, logdata = [], []
    for i in range(len(indexlist)):
        if data[i] != 0:
            logindexlist.append(indexlist[i])
            logdata.append(math.log(data[i]))
    logindexlist = np.array(logindexlist)
    logdata = np.array(logdata)
    st.markdown("first few terms:")
    st.markdown(list(data[:20]))
    with st.sidebar.form(key='sr form'):
        st.write("First Sequence")
        selected_functions = [function_set_dict[d] for d in st.multiselect('functions used in symbolic regression',options = function_set_labels,default=['x+y','x-y','x/y','x*y','-x','1/x','sqrt(x)'])]
        run_sr = st.form_submit_button('run symbolic regression using gplearn') 
        logselected_functions = [function_set_dict[d] for d in st.multiselect('functions used in symbolic regression (semilog plot)',options = function_set_labels,default=['x+y','x-y','x/y','x*y','-x','1/x','sqrt(x)'])]
        run_sr_log = st.form_submit_button('run symbolic regression on log of sequence using gplearn')   
    ylabel = 'A'+sequence1.s+'(n)'
    if run_sr:
        sr = SymbolicRegressor(function_set=selected_functions,n_jobs=-1)
        sr.fit(indexlist.reshape(-1,1),data)
        ydata = sr.predict(indexlist.reshape(-1,1))
        dfsr = pd.DataFrame(np.array([indexlist,ydata]).T)
        dfsr.columns = ['n', ylabel]
        fig1.add_trace(go.Scatter(x=indexlist,y=ydata,mode='lines',name='regression fit'))
        sr_exp = factor(simplify(expand(sympify(str(sr._program),locals=sympy_funcs))))
        if len(sr_exp.atoms(Symbol)) > 0:
            X0 = list(sr_exp.atoms(Symbol))[0]
            sr_exp = sr_exp.subs(X0,n)
        st.write("Regression function =",round_expr(sr_exp,num_digits))
    if run_sr_log:
        logsr = SymbolicRegressor(function_set=logselected_functions,n_jobs=-1)
        logsr.fit(logindexlist.reshape(-1,1),logdata)
        logydata = np.exp(logsr.predict(logindexlist.reshape(-1,1)))
        logdfsr = pd.DataFrame(np.array([logindexlist,logydata]).T)
        logdfsr.columns = ['n', ylabel]
        logfig1.add_trace(go.Scatter(x=logindexlist,y=logydata,mode='lines',name='regression fit'))
        logsr_exp = exp(factor(simplify(expand(sympify(str(logsr._program),locals=sympy_funcs)))))
        if len(logsr_exp.atoms(Symbol)) > 0:
            X0 = list(logsr_exp.atoms(Symbol))[0]
            logsr_exp = logsr_exp.subs(X0,n)
    st.plotly_chart(fig1,use_container_width=True)
    
    st.markdown("""---""")
    if run_sr_log:
        st.write("Regression function for semilog plot =",round_expr(logsr_exp,num_digits))
    st.plotly_chart(logfig1,use_container_width=True)
    summary_flag = st.sidebar.selectbox('First sequence summary',('Yes','No'),index=1,key='summary1')
    if summary_flag=='Yes':
        st.write(sequence1.OEISstruct['bfile'].describe())
    if st.selectbox('Select second sequence?',('Yes','No'),index=1)=='Yes':
        st.header("Second sequence")
        search_terms2 = st.text_input('Search terms','A'+str(seqnum2),key='search2')
        rx2 = searchregex.search(requests.get('https://oeis.org/search?q='+urllib.parse.quote_plus(search_terms2)+'&language=english&go=Search').text)
        if hasattr(rx2,'groups') and len(rx2.groups()) > 0:
            a, b = rx2.groups()
            if a == b:
                seqnum2 = int(a)
        sequence2 = OEISsequence(seqnum2)
        if 'name' not in sequence2.OEISstruct or not hasattr(rx2,'groups') or len(rx2.groups()) == 0:
            st.write("Sequence not found. Please try again.")
        else:
            st.write("OEIS sequence ",seqnum2,": ", sequence2.OEISstruct['name'])   
            fig2, logfig2 = sequence2.plot_data()
            indexlist2, data2 = sequence2.get_data_trimmed()
            logindexlist2, logdata2 = [], []
            for i in range(len(indexlist2)):
                if data2[i] != 0:
                    logindexlist2.append(indexlist2[i])
                    logdata2.append(math.log(data2[i]))
            logindexlist2 = np.array(logindexlist2)
            logdata2 = np.array(logdata2)
            st.markdown("first few terms:")
            st.markdown(list(data2[:20]))
            st.plotly_chart(fig2,use_container_width=True)
            st.plotly_chart(logfig2,use_container_width=True)
            summary_flag2 = st.sidebar.selectbox('Second sequence summary',('Yes','No'),index=1,key='summary2')
            if summary_flag2=='Yes':
                st.write(sequence2.OEISstruct['bfile'].describe())
