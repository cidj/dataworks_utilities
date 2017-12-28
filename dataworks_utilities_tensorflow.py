#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Thu Aug 17 09:35:36 2017

@author: Tao Su
Email: uku.ele@gmail.com

These are some tools or helper functions for data analysis and more written in
Python 3.
"""


import string,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from scipy.optimize import curve_fit


def drop_columns(cols,actdata,inplace=False):
    """
    Drop all the given columns if they exist.
    
    Parameters: 
        cols: A list of columns to drop.
        actdata: A dataframe.
        inplace: Whether inplace.
    
    Returns:
        actdata: A dataframe.
    """
    
    for ii in cols:
        if ii in actdata.columns:
            actdata.drop(ii,axis=1,inplace=inplace)
            
    return actdata

            
def rem_str(prelist,names):
    """
    Remove some substrings from some given strings
    
    Parameters:
        prelist: A list of the substrings.
        names: A list of strings.
    
    Returns:
        names: The processed list.
    """
    
    for prefix in prelist:
        names=[name.replace(prefix,'') for name in names]
        
    return names


def clear_columns(prefixlist,datas):
    """
    First, this function removes all the prefixes of the names of the 
    columns for the given dataframe. Sencond, it removes the underline '-', and
    convert all names into lowercase. At last, if there are duplications of 
    names, it just picks on of them randomly, and drop others. This function may
    change the names of the columns of the input dataframe.
    
    Parameters:        
        prefixlist: A list of prefixes to remove.
        datas: The input dataframe.
    
    Returns:
        datas.iloc[:, r]: The processed dataframe, which is a slice of the original
        one.
    """

    ori_columns=datas.columns.tolist()
    ccc=rem_str(prefixlist,ori_columns)
    ccc=rem_str('_',ccc)
    ccc=[c.lower() for c in ccc]
       
    d = {key: value for (key, value) in zip(ori_columns,ccc)}
    datas.rename(columns=d,inplace=True)

    u, i = np.unique(datas.columns, return_index=True)
    y=u[np.argsort(i)]   
    
    r=[datas.columns.tolist().index(rr)for rr in y]

    return datas.iloc[:, r] 


def renorm(xlis,mlis):
    """
    Use a list/array mlis to modulate the values in list/array xlis and renormed the result.
    
    Parameters:        
        xlis: List/array to be tuned.
        mlis: Weights used to tune xlis.
    
    Returns:
        A normalized array.
    """
   
#    assert(len(xlis)==len(mlis))
    return np.array(xlis)*np.array(mlis)/np.dot(xlis,mlis)[:,np.newaxis]


def classlist_from_intervals(the_intervals):
    """
    Accept a list/array of numbers and return intervals as string type.
    
    Parameters:        
        the_intervals: A list or array to be sliced.
    
    Returns:
        A string list of intervals.
    """

    return pd.Series(pd.IntervalIndex.from_breaks(np.array(the_intervals))).astype(str)


def to_class(numlist,classlist=string.ascii_lowercase):
    """
    Convert a list of numbers to a another list according to their order.
    
    Parameters:        
        numlist: A integer number list.
        classlist: Another list, letters by default.
    
    Returns:
        A list of classes.
    """

    return np.vectorize(lambda t: classlist[t])(numlist)


def proba_redefined_predict(model,X,weigh,classes=string.ascii_lowercase):
    """
    To predict a classification problem with tuned weight on the results.
    
    Parameters:       
        model: Classification model with a predict_proba method.
        X: Data with features.
        weigh: Weights' list.
        classes: classes' labels according to the order.
    
    Returns:
        predict: Tuned prediction result.
    """

    y_proba=model.predict_proba(X)
    tuned=renorm(y_proba,weigh)
    y_max_arg=tuned.argmax(axis=1)
    predict=to_class(y_max_arg,classes)
    
    return predict


def value_to_two_class(lim,val_arr,cla_arr=string.ascii_lowercase):
    """
    Compare the values to lim and return the classs according to the results.
    
    Parameters:
        lim: Number as the limit.
        val_arr: The array to be classified.
        cla_arr: The array of classes' labels.
    
    Returns:
        A list of two classes.
    """
    
    return [cla_arr[i] for i in (np.array(val_arr)>lim).astype(int)]


def value_to_class_index(bin_arr, val_arr):
    """
    Compare the values to a bin list and return the classs indices
    according to the results.
    
    Parameters:
        bin_arr: A list of numbers..
        val_arr: The array to be classified.
    
    Returns:
        Classes' indexes.
    """
#    return pd.cut(val_arr,bin_arr,labels=False)
    return np.digitize(val_arr,bin_arr,right=True)-1

def value_to_class_label(bin_arr, val_arr,cla_arr=string.ascii_lowercase):
    """
    Compare the values to a bin list and return the classs labels
    according to the results.
    
    Parameters:
        bin_arr: A list of numbers..
        val_arr: The array to be classified.
        cla_arr: The array of classes' labels.
    
    Returns:
        Classes' labels.
    """

    return [cla_arr[i] for i in value_to_class_index(bin_arr, val_arr)]


def value_to_class_interval(bin_arr, val_arr):
    """
    Compare the values to a bin list and return the intervals
    according to the results.
    
    Parameters:
        bin_arr: A list of numbers..
        val_arr: The array to be classified.
    
    Returns:
        Intervals as classes.
    """
    
    return pd.Series(pd.cut(val_arr,bin_arr)).astype(str)

def load_dataset(path_dir, filelist,numlist,dtype=None):
    """
    Load csv files according to specific numbers.
    
    Parameters:
        path_dir: Path of the files.
        filelist: File names list.
        numlist: A list of numbers descirbes the number of records to load, respectively.
        If the number is negative, it means to load the entire file.
    
    Returns:
        actdat: A dataframe.
    """
    
    actdat=pd.read_csv(os.path.join(path_dir,filelist[0]),dtype=dtype).iloc[0:numlist[0]]    
    for i in range(1,len(filelist)):
        if numlist[i]>0:
            actdat=actdat.append(pd.read_csv(path_dir+filelist[i],dtype=dtype).iloc[0:numlist[i]])
        else:
            actdat=actdat.append(pd.read_csv(path_dir+filelist[i],dtype=dtype))
    
    actdat.reset_index(drop=True,inplace=True)
    
    return actdat


def DictVectDataFrame(testdata):
    """
    Vectorize non-numeric features in dataframe while keep numeric features
    unchanged.
    
    Parameters:
        testdata: A dataframe.
    
    Returns:
        zzz: The result as a dataframe.
    """
    
    vec = DictVectorizer()
    xxx=vec.fit_transform(testdata.to_dict('records')).toarray()
    zzz=pd.DataFrame(xxx,columns=vec.feature_names_)
    
    return zzz


def feature_target_seperate(dat,cols,targ,sd=True):
    """
    Drop useless columns and seperate feature columns and targets. If there is
    no target columns, an empty series will be returned.
    
    Parameters:
        dat: Dataframe.
        cols: Columns' names to drop or to select.Integer 1 can be used for all columns.
        sd: If True, select cols, otherwise drop them.
        targ: Columns' names of targets.    
    
    Returns:
        X: Features columns.
        Y: Targets columns.
    """
    
    if targ in dat.columns:
        Y=dat[targ]
    else:
        Y=pd.Series()
    
    if sd:
        if cols==1:
            X=dat
        else:
            X=dat[cols]
    else:
        if cols==1:
            X=pd.Series()
        else:
            X=dat.drop(flatten_one([cols,targ]),axis=1)
    
    return X,Y
    

def all_same(items):
    """
    Check if all items in a list are equal.
    
    Parameters:
        items: A list.
    
    Return: 
        A bool value.
    """
    
    return all(x == items[0] for x in items) 


def brcadd(*args):
    """
    Add strings together. It allows some of the strings are lists with the same
    length, in which case other non-list strings are added into every string
    in the list, like broadcasting.
    
    Parameters:
        args: A list of strings or list of strings.
    
    Returns:
        res: A list of strings. Here the length of the list equals to the length
        of the list in args.
    """
    
    #Find the list use a mask list.
    bargs=[isinstance(i,list) for i in args]
 
    #If all the elements are strings, just connect them.
    if not any(bargs):
        return ''.join(args)
    
    #Collect the lengths of the lists and check.
    lists=[len(args[i]) for i in range(len(args)) if bargs[i]]
    assert(all_same(lists))
    length=lists[0]
    
    #Expand strings to list with the same strings such that all the lists have
    #the same length.
    lis_args=[args[i] if bargs[i] else [args[i]]*length for i in range(len(args))]
    
    #Join the strings of the same positions and get the result.
    res=[''.join([item[i] for item in lis_args]) for i in range(length)]
    
    return res


def flatten_one(list_of_lists):
    """
    Flatten a list of object/list mix. It remove the brakets of the elements if
    there are lists, and imbed them there.
    
    Parameters:
        list_of_lists: A list with lists in it.
    
    Return:
        A list generator which removed one braket-layer.
    """
    
    for x in list_of_lists:
        if isinstance(x, list):
            for y in x:
                yield y
        else:
            yield x


def flatten(lst):
    """
    Flatten a list until no list elements in it.
    
    Parameters:
        lst: A list.
    
    Return:
        A list generator.
    """
    
    for x in lst:
        if isinstance(x, list):
            for x in flatten(x):
                yield x
        else:
            yield x
            
            
def classified_weighted_assess(jid,DfA,gbs,wg,DfB,vl):
    """
    This function accept a dataframe which contains some object and corresponding
    elements, e.g. people,and their interactions as weights, and another 
    dataframe which contains some addable properties of the elements, such as earn,
    cost,time, to compute how the properties are split in a given group. This function
    may change the input dataframes.
    
    Parameters:
        jid: The column name of the elements, for example, 'userId','byerId'. Here,
        the jid set in DfA is preferred to be the same as the jid set in DfB. For
        any items in DfA but not in DfB, they will get nan values. On the other hand,
        for items in DfB but not in DfA, the last column in result DfB will get nans.        
        DfA: The dataframe contains objects~jid relations. Here is an instance.
        "candy_name", "city", "date", "buyerId","buy_times".
        gbs: Columns as groupby objects, like ['candy_name','city','date'].
        wg: The weight column, like 'buy_times'. It shold be the only reasonable
        connection between DfA and DfB, so that all the values from DfB can be 
        attributed to this tie (jie's behavior).
        DfB: The dataframe contains elements and the addable properties, e.g. some
        customers and their cost on candies.
        vl: The value property, the cost, earning, etc. It's also available for
        multi-columns in more than one properties cases, where just use a values'
        list, e.g. ['time','cost'].
    
    Returns:
        A_gbs_vl: A slice of dataframe, contains the group gbs as index, the weights
        of the group, the correlated elements number of the group and the values of
        the group.
        DfB: It's DfB but a weights correlated to the elements is added as a column.
    """
    
    #Some notes:
    ##simple count
    #yyy1=DfA[jid].groupby(DfA['adid']).count()
    
    ##connect classes to subclasses
    #yyy2=DfA[jid].groupby(DfA['adid']).unique()
    
    ##below is it: count subclasses kinds numbers
    #yyy3=DfA[jid].groupby(DfA['adid']).unique().apply(len)
    
    ##count subclasses instances numbers
    #yyy4=DfA[jid].groupby(DfA['adid']).value_counts()
    
    #yyy5=DfA.groupby(gbs).size()
    ##yyy6=DfA[jid].groupby(DfA[gbs]).size() #wrong!
    
    #Define some names for new columns.
    jid_num=brcadd(jid,'s_num')
    tot_wg=brcadd('tot_',wg,'_',jid) 
    wg_pared_jid=brcadd(jid,'_pared_'+wg) 
    vl_jid=brcadd(vl,'_',jid)
    vl_pared_jid=brcadd(jid,'_pared_',vl)
    tot_vl_pared_jid=brcadd('tot_',vl_pared_jid) 
    
    #A groupby gbs to get jid counts by gbs.
    A_jidno_gbs=DfA[jid].groupby([DfA[cl] for cl in gbs]).count().rename(jid_num)
    
    #A weights groupby uid to get total weights of uids.
    xxx=DfA[wg].groupby(DfA[jid]).sum()
    
    #Put weights in B
    #Those jid which are not in DfA may result some jid in DfB don't have weight wg.
    DfB=DfB.join(xxx,on=jid)
    DfB.rename(columns={wg:tot_wg},inplace=True) 
    
    #Merge A and B. This may cause problems if the jids of the two are not the same.
    A_B=DfA.merge(DfB,on=jid,how='left')
    A_B.rename(columns=dict(zip(vl,vl_jid)),inplace=True)
    
    #Compute the components ratios of row for the user.
    A_B[wg_pared_jid]=A_B[wg]/A_B[tot_wg]
    #Compute the corresponding value for the user.
    A_B[vl_pared_jid]=A_B[vl_jid].multiply(A_B[wg_pared_jid], axis="index")
    
    #The combination of A and B group sum by gbs to get the total wg and values by gbs.
    A_B_gbs=A_B.groupby(gbs).sum()
    A_B_gbs.rename(columns=dict(zip(vl_pared_jid,tot_vl_pared_jid)),inplace=True)
    
    #Join the weights of uids and pick the required columns.
    A_B_gbs=A_B_gbs.join(A_jidno_gbs)
    A_gbs_vl=A_B_gbs[list(flatten_one([wg,jid_num,tot_vl_pared_jid]))]
    
    return A_gbs_vl,DfB


def list_reorder(lis,dic,behind=True):
    """
    Reorder a list.
    
    Parameters:
        lis: A list.
        dic: A dictionary, both keys and values shold be in the list.
        behind: If True, rearange the values just behind the keys; if False, in front
        of it.
    
    Returns:
        A reordered list.
    """
    
    def pos(ind):
        if behind:
            return ind+1
        else:
            return ind
     
    lst=list(lis)
    for k,v in dic.items():
        lst.remove(v)
        lst.insert(pos(lst.index(k)),v)
    
    return lst


def insert_with_labels(dat,dic,behind=True):
    """
    Insert a series to a dataframe as a column, using the position of the label
    names.
    
    Parameters:
        dat: The dataframe.
        dic: A dictionary in a 'label name: series name' form. Label name indicates
        the position to insert, and series name is the name of the series as the label
        of the new column.
        behind: If True, insert the columns behind the given columns, otherwise in
        front of them.
    
    Returns:
        Dataframe.
    """
    
    def pos(ind):
        if behind:
            return ind+1
        else:
            return ind
        
    for k,v in dic.items():
        dat.insert(pos(dat.columns.tolist().index(k)),v.name,v)
        
    return dat


def features_sparseness(dat,sort=0):
    """
    Find sparse features of a dataframe. Here, the function just lists the
    sparseness of the features as a series.
    
    Parameters:
        dat: A dataframe.
        sort: How to sort the result. 0 means no sorting, 1 means ascending and
        -1 means descending.
        
    Returns:
        res: A series with features as indices and the sparseness (the portion
        of the major) as values.
    """    
        
    lblst=dat.columns.tolist()
    ll=len(dat)
    res=pd.Series(index=lblst,name='sparseness')
    
    for lb in lblst:
        ct=dat[lb].value_counts()
        res[lb]= ct.iloc[0]/ll
        
    if sort==1:
        res.sort_values(ascending=True,inplace=True)
    elif sort==-1:
        res.sort_values(ascending=False,inplace=True)
    else:
        pass
    
    return res


def save_and_replace_model(model, model_path,model_name):
    """
    Save a scikit-learn model.
    
    Parameters:
        model: Model to save.
        model_path: Directory.
        model_name: Name of the model file.
    
    Returns:
        No return.
    """
    
    model_path=model_path+'/'+model_name
    
    if os.path.isfile(model_path):
        try:
            os.remove(model_path)
            joblib.dump(model, model_path)
            print(os.path.basename(model_path)+" is replaced and saved!")
        except OSError:
            print("Models couldn't be saved")
    else:
        if os.path.isdir(os.path.dirname(model_path)):
            pass
        else:
            os.mkdir(os.path.dirname(model_path))
        joblib.dump(model, model_path)
        print(os.path.basename(model_path)+" is saved!")


def series_to_supervised(data, n_in=1, delta_in=1, n_out=1,delta_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    
    Parameters:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        delta_in: Interval of input columns.
        n_out: Number of observations as output (y).
        delta_out: Interval of output columns.
        dropnan: Boolean whether or not to drop rows with NaN values.
        
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    
    if (type(data) is list) or (type(data) is pd.Series):
         n_vars = 1
    else:
         n_vars =data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -delta_in):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out, delta_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def inverse_cumsum(y,x0): 
    """
    A inverse function of the numpy function cumsum, axis=-1 case.
    
    Parameters:
        y: A cumsum array.
        x0: The fist element of the original array, which is needed when
        reconstructing it.
        
    Returns:
        The original array.
    """
    
    return np.concatenate((np.expand_dims(x0,axis=-1),np.diff(y)),axis=-1)


def inverse_diff(y,x0):
    """
    A inverse function of the numpy function diff, axis=-1 case.
    
    Parameters:
        y: A diff array.
        x0: The fist element of the original array, which is needed when
        reconstructing it.
        
    Returns:
        The original array.
    """
    
    return np.concatenate((np.expand_dims(np.zeros(y.shape[0:-1]),axis=-1),np.cumsum(y,axis=-1)),axis=-1)+np.expand_dims(x0,axis=-1)


def diff_log(x):
    """
    First, a log function then a diff function to some kinds of time series, such as
    stocks, to make the data smooth and easy to handle. Mathematically, it coverts
    1-centered sequences to 0 centered ones and takes the differences.
    
    Parameters:
        x: The original sequences.
        
    Returns:
        Transformed sequences and the first elements of the orignial sequences,
    which may be used to do the inverse transformation.
    """
    
    return np.diff(np.log(x)),np.log(x)[0]


def inverse_diff_log(y,log0):
    """
    Inverse function of diff_log.
    
    Parameters:
        y: The transformed sequence.
        log0: The log(x)[0] in diff_log function, or its second return.
        
    Returns:
        The original sequence.
    """
    
    return np.exp(inverse_diff(y,log0))


def indices_one_hot(labels_indices, num_classes=10):
    """
    Convert class labels from scalars to one-hot vectors.
    
    Parameters:
        labels_indices: The indices of labels.
        num_classes: The number of classes.
        
    Returns:
        One hot encoded label indices.
    """
    
    num_labels = labels_indices.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_indices.ravel()] = 1
    
    return labels_one_hot


def pie_plot(data,ara,rd_f,cla_arr=string.ascii_lowercase):
    """
    Draw a pie chart using given data.
    
    Parameters:
        data: Data, an array, a list or a series.
        ara: Bin list to generate intervals.
        rd_f: String of aggregate functions, such as 'size', 'sum', etc.
        cha_arr: Labels of class array.
    
    Returns:
        No return.
    """
    
    data=pd.Series(data)
    dataclass=pd.Series(value_to_class_label(ara,data,cla_arr))
    
    parti=data.groupby(dataclass).agg(rd_f)
    
    labels=parti.index
    parts = parti.tolist()
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','peru',
              'teal','cornflowerblue','crimson','cadetblue','beige']

    plt.pie(parts, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=140)   
    plt.axis('equal')
    
    
def find_correlated(actdata,threshold=0.9,toplot=True):
    """ 
    Accept a pandas dataframe and find out the high correlated
    columns, print out the coef and plot the matrix.
    
    Parameters:        
        actdata: Input dataframe.    
        threshold: Pearson correlation coeffients which are larger than this
        value will be listed.    
        toplot: If true, plot the matrix as an image.
    
    Returns:
        No return.
    """
 
    theratio=actdata.corr()
    for ii in range(0,theratio.shape[0]):
        for jj in range(ii,theratio.shape[1]):
            if abs(theratio.iloc[ii][jj])>threshold and theratio.columns[ii]!=theratio.columns[jj]:
                print(theratio.columns[ii]+" : "+theratio.columns[jj]+": corr: "+str(theratio.iloc[ii][jj]))            

    if(toplot):            
        plt.matshow(theratio.as_matrix())
        

def plot_hist_scatter(dat, x,y,ara=None):
    """
    Just for tentatively getting a feeling about data. This function take a 
    dataframe and two columns' names and plot their scatter plot and 
    y's histogram. There are x=0 and x!=0 two cases to be plotted.
    
    Parameters:        
        dat: Input dataframe.
        x: Name of the independent variable.
        y: Name of the dependent variable.
        ara: Range for the histogram of y.
    
    Returns:
        No return.
    """
    
    dat_n0=dat[dat[x]>0]
    
    f, axarr = plt.subplots(2, 2,sharex='col')
    axarr[0, 0].hist(dat[y],ara)
    axarr[0, 0].set_title('histogram with 0s')
    axarr[0, 1].scatter(dat[x], dat[y])
    axarr[0, 1].set_title('scatter with 0s')
    axarr[1, 0].hist(dat_n0[y],ara)
    axarr[1, 0].set_title('histogram without 0s')
    axarr[1, 1].scatter(dat_n0[x], dat_n0[y])
    axarr[1, 1].set_title('scatter without 0s')
    

def plot_power_fitting(x,y,bias=True):
    """
    This function plot histograms and do the power fitting.
    
    Parameters:        
        x: Input x.
        y: Input y.
        bias: If True, bias is allowed.
    
    Returns:
        popt: The regression coefficients' array.
        pcov: The estimated covariance of popt.
    """
    
    def funcc(x, a, b, c):  
        return a * x**b +c
    
    def func0(x,a,b):
        return a * x**b
    
    if bias:
        popt, pcov = curve_fit(funcc, x, y, p0=[1.0, -1.0, 0.0]) 
        yf = [funcc(i, popt[0],popt[1],popt[2]) for i in x]
    else:
        popt, pcov = curve_fit(func0, x, y, p0=[1.0, -1.0]) 
        yf = [func0(i, popt[0],popt[1]) for i in x]
        
    plt.plot(x,y,'o')
    plt.plot(x,yf,'--')
    
    return popt,pcov

 
class BatchedDataset:
    """
    A dataset with next-batch method.
    
    Parameters:
        data: Structured data.
        is_shuffle:If true, shuffle the data every epoch.
        
    Attributes:
        _shuffle: If it is shuffled dataset, this is true. Otherwise it's False.
        _index_in_epoch: The index of the batch within the epoch.
        _epochs_num: The number of epochs completed.
        _data: The current data of that epoch.
        _num_examples: The total number of examples in the data.
        _data_order: The order of data in the epoch indicated by indexes.
        _all_data_orders: All the orders that experienced are recorded here.
        
    Methods:
        next_batch: Return a batch of samples according to batch_size.
        reset_epoch: Reset the epoch from its start.             
    """

    def __init__(self, data, is_shuffle = False):
        self._shuffle = is_shuffle
        self._index_in_epoch = 0
        self._epochs_num = 0
        self._data = data
        self._num_examples = data.shape[0]
        self._data_order=np.arange(0, self._num_examples)
        self._all_data_orders=self._data_order
    
    def __ifshuffle(self, indexes, shuf):
        if shuf:
            np.random.shuffle(indexes)
        else:
            pass
        return indexes
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start == 0 and self._epochs_num == 0:
            idx = np.arange(0, self._num_examples)
            idx = self.__ifshuffle(idx,self._shuffle)
            self._data = self._data[idx]
            self._data_order=self._data_order[idx]
            self._all_data_orders=np.c_[self._all_data_orders,self._data_order]

        if start + batch_size > self._num_examples:
            self._epochs_num += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            idx = np.arange(0, self._num_examples)
            idx = self.__ifshuffle(idx,self._shuffle)
            self._data = self._data[idx]
            self._data_order=self._data_order[idx]
            self._all_data_orders=np.c_[self._all_data_orders,self._data_order]
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end]  
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]
        
    def reset_epoch(self):
        self._index_in_epoch=0

