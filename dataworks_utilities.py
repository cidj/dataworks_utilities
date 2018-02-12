#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Thu Aug 17 09:35:36 2017

@author: Tao Su
Email: uku.ele@gmail.com

These are some tools or helper functions for data analysis and more written in
Python 3.
"""


import string, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from scipy.optimize import curve_fit
import multiprocessing
from collections import defaultdict


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


def list_duplicates_of(seq,item):
    """
    List duplicates of an item in a sequence.
    From https://stackoverflow.com/questions/5419204/

    Parameters:
        seq: The sequence.
        item: The item.

    Returns:
        locs: An indexes list of the item in seq.

    Example:
        In: source = "ABABDBAAEDSBQEWBAFLSAFB"
        In: print (list_duplicates_of(source, 'B'))
        Out: [1, 3, 5, 11, 15, 22]
    """

    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def list_duplicates(seq,threshold=1):
    """
    List duplicates of all items in a sequence.
    Modified from https://stackoverflow.com/questions/5419204/

    Parameters:
        seq: The sequence.

    Returns:
        locs: A generator for duplicates

    Example:
        In: source = "ABABDBAAEDSBQEWBAFLSAFB"
        In: list(list_duplicates(source,threshold=2))
        Out :
        [('A', [0, 2, 6, 7, 16, 20]),
         ('B', [1, 3, 5, 11, 15, 22]),
         ('D', [4, 9]),
         ('E', [8, 13]),
         ('S', [10, 19]),
         ('F', [17, 21])]
    """

    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>=threshold)


def strSeq_uniquify(strSeq,connector='_'):
    """
    Rename a sequence of strings if there are two or more of them are identical.

    Parameters:
        strSeq: The orignal sequence, probably a list or column names.
        connector: The connector between the name and the ordinal number.

    Returns:
        new_strSeq: The new list with all the same names distinguished with
        numbers.

    Example:
        In: strseq
        Out: Index(['blah', 'blah2', 'blah3', 'blah', 'blah'], dtype='object')
        In: strSeq_uniquify(strseq,connector='@')
        Out: ['blah', 'blah2', 'blah3', 'blah@0', 'blah@1']
    """

    fm="{}"+connector+"{}"

    new_strSeq = []
    for item in strSeq:
        counter = 0
        newitem = item
        while newitem in new_strSeq:
            counter += 1
            newitem = fm.format(item, counter-1)
        new_strSeq.append(newitem)

    return new_strSeq


def clear_columns(prefixlist,datas,style=0, inplace=False):
    """
    First, this function removes all the prefixes of the names of the
    columns for the given dataframe. Sencond, it removes the underline '-', and
    convert all names into lowercase. At last, if there are duplications of
    names, it just picks on of them randomly, and drop others. (Make sure they
    are equal before using this function.

    Parameters:
        prefixlist: A list of prefixes to remove.
        datas: The input dataframe.
        style: 0 means all letters are in lowercase, while 1 in uppercase and 2
        first character capitalized and the rest lowercased.
        inplace: Whether to return a new DataFrame.

    Returns:
        datas.iloc[:, r]: The processed dataframe, which is a slice of the original
        one.
    """
    func = {0: str.lower,
            1: str.upper,
            2: str.capitalize}

    ori_columns=datas.columns.tolist()
    ccc=rem_str(prefixlist,ori_columns)
    ccc=rem_str('_',ccc)
#    ccc=[c.lower() for c in ccc]
    ccc=[func[style](c) for c in ccc]

    d = {key: value for (key, value) in zip(ori_columns,ccc)}
    datas_renamed=datas.rename(columns=d,inplace=inplace)
    new_datas=datas if inplace else datas_renamed

    u, i = np.unique(new_datas.columns, return_index=True)
    y=u[np.argsort(i)]

    r=[new_datas.columns.tolist().index(rr)for rr in y]

    return new_datas.iloc[:, r]


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


def proba_redefined_predict(model,X,weigh):
    """
    To predict a classification problem with tuned weight on the results.

    Parameters:
        model: Classification model with a predict_proba method.
        X: Data with features.
        weigh: Weights' list, the order should be the same to model.classes_.

    Returns:
        predict: Tuned prediction result.
    """

    y_proba=model.predict_proba(X)
    tuned=renorm(y_proba,weigh)
    y_max_arg=tuned.argmax(axis=1)
    predict=to_class(y_max_arg,model.classes_)

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
        bin_arr: A list of numbers.
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
        dtype: The dtype in pandas.read_csv.

    Returns:
        actdat: A dataframe.
    """

    actdat=pd.read_csv(os.path.join(path_dir,filelist[0]),dtype=dtype,nrows=0)
    for i in range(0,len(filelist)):
        if numlist[i]>0:
            actdat=actdat.append(pd.read_csv(path_dir+filelist[i],dtype=dtype,nrows=numlist[i]))
        else:
            actdat=actdat.append(pd.read_csv(path_dir+filelist[i],dtype=dtype))

    actdat.reset_index(drop=True,inplace=True)

    return actdat


def append_to_csv(df, csvFilePath, sep=",", supersede=False):
    """
    Append a dataframe to a csv file.

    Parameters:
        df: The dataframe.
        csvFilePath: The csv file path.
        sep: Separator of the file.
        supersede: If true, replace the existing file.

    Returns:
        No return.

    Example:
        In: df1
        Out:
          A_Id  P_Id   CN1   CN2   CN3
        0  AAA   333   710   750   750
        1  BBB   444   180   734   778
        2  BBB   222  1727  1734  1778
        3  AAA   222  1727  1734  1778
        4  XXX   222  1727  1734  1778

        In: df2
        Out:
            CN1   CN2 A_Id  P_Id   CN3
        0   710   750  AAA   333   750
        1   180   734  BBB   444   778
        2  1727  1734  BBB   222  1778
        3  1727  1734  AAA   222  1778
        4  1727  1734  XXX   222  1778

        In: append_to_csv(df1,'new.csv')
        #new.csv file content:
        A_Id	P_Id	CN1	CN2	CN3
        AAA	333	710	750	750
        BBB	444	180	734	778
        BBB	222	1727	1734	1778
        AAA	222	1727	1734	1778
        XXX	222	1727	1734	1778

        In: append_to_csv(df2,'new.csv')
        #new.csv file content:
        A_Id	P_Id	CN1	CN2	CN3
        AAA	333	710	750	750
        BBB	444	180	734	778
        BBB	222	1727	1734	1778
        AAA	222	1727	1734	1778
        XXX	222	1727	1734	1778
        AAA	333	710	750	750
        BBB	444	180	734	778
        BBB	222	1727	1734	1778
        AAA	222	1727	1734	1778
        XXX	222	1727	1734	1778
    """

    if (not os.path.isfile(csvFilePath)) or supersede==True:
        df.to_csv(csvFilePath, index=False, sep=sep)

    else:
        d_od=df.columns
        f_od=pd.read_csv(csvFilePath,nrows=0,sep=sep).columns
        if np.setxor1d(d_od,f_od).size:
            raise Exception("Columns do not match: Dataframe columns are: ",
                            d_od, ". CSV file columns are: ", f_od, ".")

        else:
            df[f_od].to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def csvs_scattered_to_grouped(path_dir, inlist, outlist, gcols,
                              sort=1, scols=None, catalog="", supersede=False):
    """
    A function to combine csv files where some values of specific columns are scattered,
    and group them, then save them into seperate files where all the same groups are
    saved in the same files. It is useful to do this when dealing with large files in
    limited memory. This function takes time but make available to analyse big
    file through splitting it into smaller files.

    Parameters:
        path_dir: File directory.
        inlist: Input file list.
        outlist: Output file list.
        gcols: The columns to use to group.
        sort: If 1, sort by gcols in ascending order, -1 in descending order,
        otherwise unsorted.
        scols: The columns list to use. If it is None, all columns are used. If
        not None, it must include all items in gcols.
        catalog: If it is a string, a catalog file with the name will be created.
        If False, no catalog created. Make sure that there is no column named
        '_@_FILE_', or there might be some error due to the same name.
        supersede: If True, existing files with the same names will be replaced.

    Returns:
        No return.

    Example:
        File 1: 'new1.csv'
            A_Id	P_Id	CN1	CN2	CN3
            AAA	333	710	750	750
            BBB	444	180	734	778
            BBB	222	1727	1734	1778
            AAA	222	1727	1734	1778
            XXX	222	1727	1734	1778
        File 2: 'new2.csv'
            A_Id	P_Id	CN1	CN2	CN3
            AAA	333	710	750	750
            BBB	444	180	734	778
            BBB	222	1727	1734	1778
            AAA	222	1727	1734	1778
            XXX	222	1727	1734	1778
            YYY	222	1727	1734	1778

        In: path_dir='/some/path/'
        In: inlist= ['new'+str(i+1)+'.csv' for i in range(0,2)]
        In: outlist=['out'+str(i+1)+'.csv' for i in range(0,2)]
        In: gcols=[ 'A_Id','P_Id' ]
        In: scols=['A_Id','P_Id','CN1', 'CN2']
        In: csvs_scattered_to_grouped(path_dir, inlist, outlist, gcols,
                              sort=1, scols=scols, catalog='cat.csv',supersede=True)

        The generated files are:
        'out1.csv'
        A_Id	P_Id	CN1	CN2
        AAA	222	1727	1734
        AAA	333	710	750
        BBB	222	1727	1734
        AAA	222	1727	1734
        AAA	333	710	750
        BBB	222	1727	1734
        'out2.csv'
        A_Id	P_Id	CN1	CN2
        BBB	444	180	734
        XXX	222	1727	1734
        BBB	444	180	734
        XXX	222	1727	1734
        YYY	222	1727	1734
        'cat.csv'
        A_Id	P_Id	_@_FILE_
        AAA	222	new1.csv
        AAA	333	new1.csv
        BBB	222	new1.csv
        BBB	444	new2.csv
        XXX	222	new2.csv
        YYY	222	new2.csv
    """

    filelist=[os.path.join(path_dir,i) for i in inlist]
    n_split=len(outlist)

    pdfs=pd.read_csv(filelist[0],usecols=gcols)
    pdfs.drop_duplicates(inplace=True)

    print("csvs_scattered_to_grouped: Collecting items for group.\n")
    for i in range(1,len(filelist)):
        pdfs=pdfs.append(pd.read_csv(filelist[i],usecols=gcols),ignore_index=True)
        pdfs.drop_duplicates(inplace=True)

    if sort==1:
        pdfs.sort_values(gcols,inplace=True, ascending=True)
    elif sort==-1:
        pdfs.sort_values(gcols,inplace=True, ascending=False)

    aa_ed=np.array_split(pdfs, n_split)

    if supersede:
        for i in outlist:
            if os.path.isfile(os.path.join(path_dir,i)):
                os.remove(os.path.join(path_dir,i))
            if os.path.isfile(os.path.join(path_dir,str(catalog))):
                os.remove(os.path.join(path_dir,str(catalog)))

    print("csvs_scattered_to_grouped: Start processing files:\n")
    for i in range(0,len(filelist)):
        fi=pd.read_csv(filelist[i],usecols=scols)
        for j,ja in enumerate(aa_ed):
            wrtj=pd.merge(ja, fi, how='inner', on=gcols)
            append_to_csv(wrtj, os.path.join(path_dir,outlist[j]))
        print('csvs_scattered_to_grouped: '+str(i)+' file(s) finished.')

    if catalog:
        for i, d in enumerate(aa_ed):
            d['_@_FILE_']=outlist[i]
            append_to_csv(d, os.path.join(path_dir,str(catalog)))
        print('csvs_scattered_to_grouped: Catalog file created.')


def all_equal(iterator):
    """Check if a sequence or other container contains identical element. The
    code is from http://stackoverflow.com/q/3844948/.

    Parameters:
        iterator: A sequence or other iterable object.

    Returns:
        If all elements are identical, return True, otherwise False.
    """

    iterator = iter(iterator)

    try:
        first = next(iterator)
    except StopIteration:
        return True

    return all(first == rest for rest in iterator)


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

    Example:
        In: brcadd('xy',['a','bc','d'],'b')
        Out: ['xyab', 'xybcb', 'xydb']
    """

    #Find the list use a mask list.
    bargs=[isinstance(i,list) for i in args]

    #If all the elements are strings, just connect them.
    if not any(bargs):
        return ''.join(args)

    #Collect the lengths of the lists and check.
    lists=[len(args[i]) for i in range(len(args)) if bargs[i]]

#    assert(all_equal(lists))
    if (not all_equal(lists)):
        raise ValueError('All args must have the same length!')

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
        yyy: It's a two-column dataframe contains jid and the sum number of weights
        that connect to it.

    Example:
        Suppose there are two dataframes. One (DfA) contains the ads (adid),operating
        system (os), viewtimes of ads (viewtimes), ids of users (uid), and their
        country (country). The other (DfB) contains ids of users (uid), the
        predicted class the users belong to (class_predict)  and the predicted
        values of the users (value_predict). Here we want to compare how much value
        every ['adid','country','os'] have brought, according to the
        value of the users they brought. Since every user might view more than
        one ads, so her value should be split to these ads, and we use 'viewtimes'
        as the weights here. Then we can use this function as follows:
        Here are the input dataframes:

        In: DfA
        Out:
           adid       os  viewtimes uid country
        0     u  android          2   a      GB
        1     v       os          1   b      US
        2     w  android          1   b      GB
        3     x  android          2   a      GB
        4     y       os          1   c      US
        5     v  android          3   c      GB
        6     u  android          1   a      GB
        7     x  android          1   b      AE
        8     y       os          2   a      US
        9     u  android          1   b      GB
        10    v  android          1   b      AE
        11    w  android          2   a      US
        12    x       os          1   c      AE
        13    v       os          1   d      US

        In: DfB
        Out:
          uid class_predict  value_predict
        0   b           cla              5
        1   c           clb              1
        2   d           clc              8

        In: jid='uid'
        In: gbs=['adid','country','os']
        In: wg='viewtimes'
        In: vl=['value_predict']

        In: A_gbs_vl,yyy=classified_weighted_assess(jid,DfA,gbs,wg,DfB,vl)

        In: A_gbs_vl
        Out:
                              viewtimes  uids_num  tot_uid_pared_value_predict
        adid country os
        u    GB      android          4         3                          1.0
        v    AE      android          1         1                          1.0
             GB      android          3         1                          0.6
             US      os               2         2                          9.0
        w    GB      android          1         1                          1.0
             US      android          2         1                          NaN
        x    AE      android          1         1                          1.0
                     os               1         1                          0.2
             GB      android          2         1                          NaN
        y    US      os               3         2                          0.2

        In: yyy
        Out:
          uid  tot_viewtimes_uid
        0   b                  5
        1   c                  5
        2   d                  1
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
    #Those jid which are not in DfA may result in some jids in DfB don't have weight wg.
    DfBB=DfB.join(xxx,on=jid)
    DfBB=DfBB.rename(columns={wg:tot_wg})

    #Merge A and B. This may cause problems if the jids of the two are not the same.
    A_B=DfA.merge(DfBB,on=jid,how='left')
    A_B=A_B.rename(columns=dict(zip(vl,vl_jid)))

    #Compute the components ratios of row for the user.
    A_B[wg_pared_jid]=A_B[wg]/A_B[tot_wg]
    #Compute the corresponding value for the user.
    A_B[vl_pared_jid]=A_B[vl_jid].multiply(A_B[wg_pared_jid], axis="index")

    #The combination of A and B group sum by gbs to get the total wg and values by gbs.
    A_B_gbs=A_B.groupby(gbs).sum()
    A_B_gbs=A_B_gbs.rename(columns=dict(zip(vl_pared_jid,tot_vl_pared_jid)))

    #Join the weights of uids and pick the required columns.
    A_B_gbs=A_B_gbs.join(A_jidno_gbs)
    A_gbs_vl=A_B_gbs[list(flatten_one([wg,jid_num,tot_vl_pared_jid]))]
#    yyy=DfB.loc[:, [jid]].join(xxx,on=jid)
    yyy=DfBB[[jid,tot_wg]]

    return A_gbs_vl,yyy


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

    def _pos(ind):
        if behind:
            return ind+1
        else:
            return ind

    lst=list(lis)
    for k,v in dic.items():
        lst.remove(v)
        lst.insert(_pos(lst.index(k)),v)

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

    def _pos(ind):
        if behind:
            return ind+1
        else:
            return ind

    for k,v in dic.items():
        dat.insert(_pos(dat.columns.tolist().index(k)),v.name,v)

    return dat


def save_obj_joblib(obj, obj_path,obj_name,supersede=False):
    """
    Save an object using joblib.dump.

    Parameters:
        obj: The object to save.
        obj_path: Directory.
        obj_name: Name of the object file.

    Returns:
        No return.
    """

    obj_path=os.path.join(obj_path,obj_name)

    if os.path.isfile(obj_path):
        if supersede:
            try:
                os.remove(obj_path)
                joblib.dump(obj, obj_path)
                print("save_obj_joblib: "+os.path.basename(obj_path)+" is replaced and saved!")
            except OSError:
                print("save_obj_joblib: Object couldn't be saved")
        else:
            raise OSError("save_obj_joblib: There exists a object with the same name already.")
    else:
        if os.path.isdir(os.path.dirname(obj_path)):
            pass
        else:
            os.mkdir(os.path.dirname(obj_path))
        joblib.dump(obj, obj_path)
        print("save_obj_joblib: "+os.path.basename(obj_path)+" is saved!")


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
        return agg.dropna()
    else:
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

    Example:
        In: indices_one_hot(np.array([1,2,3]),4)
        Out:
            array([[ 0.,  1.,  0.,  0.],
                   [ 0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  1.]])
    """

    num_labels = labels_indices.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_indices.ravel()] = 1

    return labels_one_hot


def get_subdataframe(Acol,Bdf):
    """
    Use colmns from a dataframe according to a name list to form a new dataframe.
    If a column does not exit in the dataframe, the values are filled with nans.
    The columns' order is the same as that of the list.

    Parameters:
        Acol: A column name list.
        Bdf: A dataframe which contains the columns.

    Returns:
        A new dataframe.

    Example:
        In: df = pd.DataFrame([[5, 6,9,10], [7, 8,4,3]], columns=list('BADE'))
        In: df
        Out:
           B  A  D   E
        0  5  6  9  10
        1  7  8  4   3
        In: setdiff_dataframe(['A','B','C'],df)
        Out:
           A  B   C
        0  6  5 NaN
        1  8  7 NaN
    """

    cAdf=pd.DataFrame(index=Bdf.index)
    for i in Acol:
        cAdf[i]=Bdf[i] if i in Bdf.columns else np.nan
    return cAdf


def dataframe_diff(xxa,xxb):
    """
    This function find the differences of two dataframes. Make
    sure there is no column named '_merge'.

    Parameters:
        xxa, xxb: Two dataframes to be compared.

    Returns:
        diff: A dataframe shows the differences of the inputs.
    """

    xa=pd.DataFrame(xxa)
    xb=pd.DataFrame(xxb)
    merged = xa.merge(xb, indicator=True, how='outer')

    diff=merged[merged['_merge'] != 'both']

    return diff


def _apply_df(args):
    df, func, num, kwargs = args
    return num, df.apply(func, **kwargs)
def apply_by_multiprocessing(df0,func,workers=4,**kwargs):
    """
    Multiprocessing 'apply' for dataframes. Main idea of this function is from:
        https://gist.github.com/tejaslodaya/562a8f71dc62264a04572770375f4bba

    Parameters:
        df: A dataframe.
        func: The function to apply.
        **kwargs: other aguments, like axis. It also should contain workers which
        determines the number of works in processing.

    Returns:
        A dataframe or series.

    Example:
        In: df = pd.DataFrame({'a':range(10), 'b':range(10,20)})
        In: df
        Out:
           a   b
        0  0  10
        1  1  11
        2  2  12
        3  3  13
        4  4  14
        5  5  15
        6  6  16
        7  7  17
        8  8  18
        9  9  19

        In: apply_by_multiprocessing(df, sum, workers=4, axis=1)
        Out:
        0    10
        1    12
        2    14
        3    16
        4    18
        5    20
        6    22
        7    24
        8    26
        9    28
        dtype: int64

        In: apply_by_multiprocessing(df, sum, workers=4, axis=0)
        Out:
        a     45
        b    145
        dtype: int64

        In: def square(x):
                return x**2
        In: apply_by_multiprocessing(df, square, workers=4)
        Out:
            a    b
        0   0  100
        1   1  121
        2   4  144
        3   9  169
        4  16  196
        5  25  225
        6  36  256
        7  49  289
        8  64  324
        9  81  361

    """

    flag=0

    if 'axis' in kwargs.keys():
        axis=kwargs['axis']
        if(axis==0):
            flag=1
            kwargs['axis']=1
            df=df0.transpose()
        else:
            df=df0
    else:
        df=df0

    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, i, kwargs) for i,d in enumerate(np.array_split(df, workers))])
    pool.close()

    result=sorted(result,key=lambda x:x[0])
    res=pd.concat([i[1] for i in result])

    if flag==1:
        return res.transpose()
    else:
        return res


def mgrid_box(X,axis=0,linj=200j,marg_rate=0.1):
    """
    Get a mesh grid range which covers all the n-dimentional points.

    Parameters:
        X: An 2-D array, consists of vectors.
        axis: if 0, the shape of every vector in X is (-1,1), otherwise (1,-1).
        linj: If it is a complex number, it is the number of points to create
        between the maximum and minimum values. Or if it is a real number, it is
        the step length between them.
        marg_rate: It shows how much of the margins should be included.

    Returns:
        zz: An array includes n subarrays. Here n is the space dimension. Each
        subarray contains coordinates of an axis.

    Example:
        In: X
        Out:
        array([[4, 5],
               [6, 7]])
        In: mgrid_box(X,axis=0,linj=4j,marg_rate=0.1)
        Out:
        array([[[ 3.8,  3.8,  3.8,  3.8],
                [ 4.6,  4.6,  4.6,  4.6],
                [ 5.4,  5.4,  5.4,  5.4],
                [ 6.2,  6.2,  6.2,  6.2]],

               [[ 4.8,  5.6,  6.4,  7.2],
                [ 4.8,  5.6,  6.4,  7.2],
                [ 4.8,  5.6,  6.4,  7.2],
                [ 4.8,  5.6,  6.4,  7.2]]])
    """

    if len(X.shape)!=2:
        raise Exception('Only 2-D array is acceptable!')

    o_axis=1-axis

    X_max=np.max(X,axis=axis)
    X_min=np.min(X,axis=axis)
    lenX=X_max-X_min
    Marg_X=marg_rate*lenX
    X_edgmin, X_edgmax = X_min - Marg_X, X_max + Marg_X

    num=np.size(X,axis=o_axis)

    slices=[slice(X_edgmin[i],X_edgmax[i],linj) for i in np.arange(num)]
    zz=np.mgrid[slices] if axis==0 else np.array([i.T for i in np.mgrid[slices]])

    return zz


def corresponding_ravel(X,axis=0):
    """
    Given an n-dimentional array, remove its 0th dimention, ravel all other dimentions,
    and return an 2-D matrix.

    Parameters:
        X: The input array.
        axis: if it is 0, the output contains columns, otherwise rows.

    Returns:
        t1: A 2-D array.

    Example:
        In: aa=np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
        In: aa
        Out:
        array([[[0, 1],
                [2, 3]],

               [[4, 5],
                [6, 7]]])
        In: Corresponding_ravel(aa,axis=0)
        Out:
        array([[0, 4],
               [1, 5],
               [2, 6],
               [3, 7]])
    """

    t1=np.c_[tuple([di.ravel() for di in X])]

    if axis==1:
        t1=t1.T

    return t1


def plot_region_prediction(X,model,linj=200j,marg_rate=0.1,centroids=None):
    """
    Plot a prediction 2-D figure using a classification or clustering model.

    Parameters:
        X: The input data, contains two columns.
        model: The model, wich should have a 'predict' method.
        linj: Number of grids in each dimension. If it is a complex number, it
        is the number of points to create between the maximum and minimum values.
        Or if it is a real number, it is the step length between them.
        marg_rate: It shows how much of the margins should be included.
        centroids: If it's clustering, you can add some centroids to plot. 2-col
        array.

    Returns:
        fig: A figure.
    """

    zz=mgrid_box(X,axis=0,linj=linj,marg_rate=marg_rate)
    zz1=corresponding_ravel(zz,axis=0)
    Z = model.predict(zz1)
    Z = Z.reshape(zz[0].shape)

#    fig=plt.figure()

    plt.imshow(Z, interpolation='nearest',
               extent=(zz[0].min(), zz[0].max(), zz[1].min(), zz[1].max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(X[:,0],X[:,1], 'k.', markersize=5)

    if centroids is not None:
#        plt.scatter(centroids[:, 0], centroids[:, 1],
#                    marker='x', s=169, linewidths=3,
#                    color='w', zorder=10)
        cs=model.predict(centroids)
        for i in range(0,len(centroids)):
            plt.text(centroids[i, 0], centroids[i, 1],cs[i],
                     fontsize=16,color='w',zorder=10)
#    return fig


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


def connect_array_by_last_dim(*data):
    """
    Connect several arrays to an array. All the arrays in the input must
    have the same shape except the last dimension. The dtypes may change, such
    as int to float, bool to int or float. Don't use other non-numerical data
    types.

    Parameters:
        data: Some arrays to be connected

    Returns:
        adata: An array, which is a connection of the inputs by the last dimension.
        bx: A vector/array of the indexes range of the input arrays in adata.
        atype: A list of the dtypes of the input arrays.

    Example:
        In: a=np.array([[1,2],[3,4]])
        In: b=np.array([[5.0],[6.0]])
        In: c=np.array([[True],[False]])
        In: connect_array_by_last_dim(a,b,c)
        Out: (array([[ 1.,  2.,  5.,  1.],
                     [ 3.,  4.,  6.,  0.]]),
            array([0, 2, 3, 4]),
            [dtype('int64'), dtype('float64'), dtype('bool')])
    """

    aax=list(map(lambda x: x.shape[:-1], data))
    if (not all_equal(aax)):
        raise ValueError('All inputs must have the same shape except the last dimension!')

    ax=list(map(lambda x: x.shape[-1], data))
    bx=np.concatenate([[0],np.cumsum(ax)])

    atype=list(map(lambda x: x.dtype, data))

    adata=np.concatenate(data,axis=-1)

    return adata,bx,atype


def separate_array_by_last_dim(adata,bx,atype):
    """
    It's the inverse function of connect_array_by_last_dim. The function accepts an array, a vector
    of indexes of subarrays and types of subarrays and separate the input array
    to several arrays according to the indexes and types.

    Parameters:
        adata: An array contains all subarrays, which are connected by the last dimension.
        bx: A vector/array of the indexes range of the subarrays in adata.
        atype: A list of the dtypes of the subarrays.

    Returns:
        cx: Array(s) which are subarrays of adata.

    Example:
        In: adata=np.array([[ 1.,  2.,  5.,  1.],
                     [ 3.,  4.,  6.,  0.]])
        In: bx=np.array([0, 2, 3, 4])
        In: atype=[np.dtype('int64'), np.dtype('float64'), np.dtype('bool')]
        In: separate_array_by_last_dim(adata,bx,atype)
        Out: (array([[1, 2],
                     [3, 4]]),
            array([[ 5.],
                   [ 6.]]),
            array([[ True],
                   [False]], dtype=bool))
    """

    if(len(atype)==1):
        cx=adata
    else:
        cx=tuple([adata[...,slice(bx[i],bx[i+1])].astype(atype[i]) for i in np.arange(0,len(bx)-1)])

    return cx


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
        _data_indexes: A vector/array of the indexes range of the input arrays in _data.
        _data_type: A list of the dtypes of the input arrays.
        _num_examples: The total number of examples in the data.
        _data_order: The order of data in the epoch indicated by indexes.
        _all_data_orders: All the orders that experienced are recorded here.

    Methods:
        next_batch: Return a batch of samples according to batch_size.
        reset_epoch: Reset the epoch from its start.
    """

    def __init__(self, *data, is_shuffle = False):
        self._shuffle = is_shuffle
        self._index_in_epoch = 0
        self._epochs_num = 0
        self._data, self._data_indexes, self._data_type = connect_array_by_last_dim(*data)
        self._num_examples = self._data.shape[0]
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
            res = np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            res = self._data[start:end]

        return separate_array_by_last_dim(res, self._data_indexes, self._data_type)

    def reset_epoch(self):
        self._index_in_epoch=0


def pd_parse_column(ser,parse_fun,pick_row=0,workers=6):
    """
    If a column of some dataframe (or a series) contains some formatted text, which can be parsed using some
    parse function (or functions written using regular expressions), then this fuction can collect
    all the result parsed from the column cells to a dataframe or a dictionary of dataframes.
    A dataframe in a row may contain duplicated columns' names and they are renamed with suffix '_0','_1',...
    Due to the dataframes in different rows may have different columns, the columns of result dataframes
    are the uninon of them.

    Parameters:
        ser: A series, it could be a column of some dataframe.
        parse_fun: A parse function which can parse the text in a cell and convert it to a dataframe.
        pick_row: The index of the row which belongs to the parse_fun created dataframe and is meant
            to be collected. When it is None, all rows will be collected to different dataframes.
        workers: The number of processes when applying the parse function parallelly.

    Returns:
        ddf: A dataframe or a dictionary of dataframes.
    """

    print('pd_parse_column: Started.')

    mapto=apply_by_multiprocessing(ser,parse_fun,workers=workers)
    print('pd_parse_column: Applied ', str(parse_fun.__name__),'.')

    class NotSame(Exception):
        def __init__(self):
            Exception.__init__(self,"pd_parse_column: The contents in the rows are not the same.")

    row_lst=[]
    for i in range(0,len(mapto)):
        mapto.iloc[i].columns=strSeq_uniquify(mapto.iloc[i].columns)
        if not mapto.iloc[i].equals(pd.DataFrame()):
            if row_lst:
                if row_lst!=mapto.iloc[i].index.tolist():
                    raise NotSame()
            else:
                row_lst=mapto.iloc[i].index.tolist()

    print('pd_parse_column: Renamed duplicated names.')

    mapto_cols=set(flatten(mapto.apply(lambda x: x.columns.tolist()).tolist()))
    print('pd_parse_column: Columns collected.')

    ddf=pd.DataFrame(columns=mapto_cols)

    def _concat_lst(pr):
        ddlst=[ddf]+[mapto.iloc[i].iloc[[pr]].rename(index={mapto.iloc[i].iloc[[pr]].index[0]: mapto.index[i]})
        if not mapto.iloc[i].empty else pd.DataFrame(index=[mapto.index[i]])
               for i in range(0,len(mapto))]
        return pd.concat(ddlst)

    if pick_row is None:
        ddf_lst=[_concat_lst(i) for i in range(0,len(row_lst))]
        ddf=dict(zip(row_lst,ddf_lst))
    else:
        ddf=_concat_lst(pick_row)
    print('pd_parse_column: Connected the results. Finished.')

    return ddf


def supervised_count_feature(s1,s0):
    """
    If you have two classes labeled 1 (abnormal) and 0 (normal), and one of their
    features is composed of some categories, you can use this function to find the
    ratios of the instances' counts between the two classes, therefore it shows
    some differences of the feature. However, due to the distribution here may not
    reflect the real feature, this method some times will cause overfitting. So
    becareful.

    Parameters:
        s1: A feature series contains some categories. All the items/rows here
        are labeled 1.
        s0: A feature series contains some categories. All the items/rows here
        are labeled 0.

    Returns:
        ss: A series, which contains the percentages of the sample counts of 1 class.
    """
    a1=s1.groupby(s1).count()
    a0=s0.groupby(s0).count()
    b0,b1=a0.align(a1)
    c1=b1.fillna(0)
    c0=b0.fillna(0)
    ss=(c1/(c0+c1))
    return ss

def supervised_add_count(ser,marker):
    """
    In supervised feature processing, use supervised_count_feature function to get
    a feature column as a series. This may cause overfitting.

    Parameters:
        ser: The series that to be processed.
        marker: A series of markers where there are 1s and 0s.

    Returns:
        new_ser: A series as a feature column.
        ss: A series which is the distribution map.
    """
    ss=supervised_count_feature(ser[marker==1],ser[marker==0])
    new_ser=ser.map(ss)
    return new_ser,ss


# Deprecated:


# def DictVectDataFrame(testdata):
#     """
#     2018/01/15: Do not use this function. Use the buit in function
#     pandas.get_dummies instead.
#
#     Vectorize non-numeric features in dataframe while keep numeric features
#     unchanged.
#
#     Parameters:
#         testdata: A dataframe.
#
#     Returns:
#         zzz: The result as a dataframe.
#     """
#
#     from sklearn.feature_extraction import DictVectorizer
#     vec = DictVectorizer()
#     xxx=vec.fit_transform(testdata.to_dict('records')).toarray()
#     zzz=pd.DataFrame(xxx,columns=vec.feature_names_)
#
#     return zzz
