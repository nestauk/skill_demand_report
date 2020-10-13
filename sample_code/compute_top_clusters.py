#!/usr/bin/env python
# coding: utf-8
'''
Description.

The purpose of this script is to compute the most representative skill cluster
from the second interation of Nesta's skills taxonomy to describe each individual
job advert. This step is necessary to build the crosswalk from occupations to skill
categories.

The most representative cluster is obtained by taking the cluster with highest
weighted average semantic similarity between each candidate cluster and all the
skills referenced in the job advert. We use a “discount factor”, which is inversely
proportional to how often a skill cluster appeared in the whole job advert dataset,
and then also a "context factor", which considers whether a large proportion of
skills mentioned in an advert belonged to the same higher level skill cluster,
to weigh the raw average similarities. We take two most representative clusters
(which can be the same): one considering the discount factor only and one
considering both the discount and the context factor. Note that 'soft skills' are
not used to compute the most representative cluster.

Finally, note that this script is computationally very intensive.

Author: Stef Garasto
'''

# # Imports and functions
# imports
import argparse
from collections import Counter, OrderedDict
from copy import deepcopy
import gzip
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity
import sys
from time import time as tt
from tqdm import tqdm

from textkernel_load_utils import tk_params, data_folder, create_tk_import_dict, read_and_append_chunks
from utils_general import TaskTimer, print_elapsed, nesta_colours,sic_letter_to_text, flatten_lol, printdf
from utils_nlp import lemmatise, highest_similarity_threshold
from utils_skills_clusters import taxonomy_2_0
from utils_skills_matching import model, generate_embeddings, generate_cluster_embeddings


DATA_PATH = Path(data_folder).parent
print(DATA_PATH)

timer = TaskTimer()
print('Done')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--fstart", help="Number of first file to process",
    default= 0, type= int)
parser.add_argument("--fend", help="Number of last file to process",
    default = 1, type= int)
args = parser.parse_args()

f_start = args.fstart
f_end = args.fend


# Load the skills taxonomy
taxonomy_2_0.skills_taxonomy_full = taxonomy_2_0.skills_taxonomy_full.set_index('preferred_label'
                                            ).loc[taxonomy_2_0.skill_list].reset_index()

# # Define functions
def flattencolumns(df, cols = ['alt_labels']):
    ''' spread a column of lists across multiple columns '''
    df1 = pd.concat([pd.DataFrame(df[x].values.tolist()).add_prefix(x)
                    for x in cols], axis=1)
    return pd.concat([df1, df.drop(cols, axis=1)], axis=1)

def join_skill_columns(data_df_small, SOFT= False):
    ''' Join all the skills columns from textkernel together. TODO:check'''
    if SOFT:
        joined_skills = data_df_small.soft_skills.fillna(''
                    ) + ';' + data_df_small.professional_skills.fillna(''
                    ) + ';' + data_df_small.it_skills.fillna(''
                    ) + ';' + data_df_small.language_skills.fillna('')
    else:
        joined_skills = data_df_small.professional_skills.fillna(''
                    ) + ';' + data_df_small.it_skills.fillna(''
                    ) + ';' + data_df_small.language_skills.fillna('')
    return tmp


def unroll_column_of_list(df, col_to_unroll, id_cols):
    ''' It takes a column where each entry is a list and spreads that list across many columns.
    It creates as many columns as the maximum number of elements in the lists'''
    assert(isinstance(id_cols,list))
    assert(isinstance(col_to_unroll,str))

    flattened_df = flattencolumns(df[[col_to_unroll] + id_cols], cols = [col_to_unroll])

    cols_to_unroll = [t for t in flattened_df.columns if col_to_unroll in t]

    return unroll_df(flattened_df, cols_to_unroll, id_cols, shared_col = col_to_unroll)

def unroll_df(df, cols_to_unroll, id_cols, shared_col = 'var'):
    ''' Takes the group of columns produced by "unroll_column_of_list" and concatenates them along the rows.
    It uses the information in "id_cols" as the identifier for the unrolled entries'''
    assert(len(df.columns) == (len(cols_to_unroll) + len(id_cols)))
    df_list = []
    for col in cols_to_unroll:
        df_list.append(df[[col] + id_cols].dropna().rename(columns = {col: shared_col})) #.set_index(id_cols))
    return pd.concat(df_list)

def clean_skill_vector(x):
    ''' Cleans extra punctuation marks after joining skill columns from Textkernel'''
    x = x.lower().strip(';')
    return re.sub(';;+',';', x)

def split_skill_vector(x):
    ''' Splits string of skills into a list of skills. Eliminates empty elements'''
    x = [t.lower() for t in x.split(';') if len(t)]
    return x

def join_skill_vector(x):
    ''' join list of skills into a string'''
    return ';'.join(x)

def clean_skill_vector2(x):
    ''' Cleans up string with skills by concatenating a split and a join operation'''
    x = join_skill_vector(split_skill_vector(x))
    return x

def resample_soc_to_n_digits(soc_code,n=3):
    """ from 4-digit SOC code to n-digit """
    if np.isnan(soc_code):
        return np.nan
    else:
        m = {1: 1000, 2: 100, 3: 10}[n]
        return (soc_code - soc_code%m)/m

def resample_soc_to_n_digits_df(soc_code_df,n=3):
    """ from 4-digit SOC code to n-digit for a dataframe"""
    #soc_code_df = soc_code_df.fillna(0)
    m = {1: 1000, 2: 100, 3: 10}[n]
    return (soc_code_df - soc_code_df%m)/m


# set up some relevant variables
columns_list = [f'profession_soc_code_{t}' for t in ['value','3','2','1']] + ['job_id']
ks_soc = ['4','3','2','1']
ks_esco = ['level_3','level_2','level_1','label_level_3','label_level_2','label_level_1']

print('done')


# Load datasets

## Load ESCO-based skills taxonomy clusters
### reload the esco list, including skills that have not been clustered
res_folder_local = '/path/to/interim/results'

# Load full clusters
esco_clusters_file = taxonomy_2_0.main_file
esco_clusters_dir = taxonomy_2_0.main_dir
esco_clusters = pd.read_csv(esco_clusters_file)
# make alt labels list
esco_clusters['alt_labels'] = esco_clusters.alt_labels.map(
        lambda x: x.split('\n') if isinstance(x,str) else [])

# adjustments to the labels
esco_clusters.loc[esco_clusters.level_2==20.0,'label_level_2'] = 'land transport (rail)'
esco_clusters.loc[esco_clusters.level_3==191.0,'label_level_3'
                 ] = 'leather production (manufacturing)'
esco_clusters.loc[esco_clusters.level_3==190.0,'label_level_3'] = 'footwear design'
esco_clusters.loc[esco_clusters.level_3==57.0,'label_level_3'] = 'marketing (branding)'

print('All ESCO skills: ', len(esco_clusters))
print('Valid ESCO skills: ',len(taxonomy_2_0.skills_taxonomy_full))

print(esco_clusters.columns)

## Load labels for ESCO-based clusters
# load cluster labels
cluster_labels_1 = pd.read_csv(f"{esco_clusters_dir}/ESCO_Essential_clusters_Level_1.csv")
cluster_labels_2 = pd.read_csv(f"{esco_clusters_dir}/ESCO_Essential_clusters_Level_2.csv")
cluster_labels_3 = pd.read_csv(f"{esco_clusters_dir}/ESCO_Essential_clusters_Level_3.csv")

cluster_labels_3 = cluster_labels_3.set_index('level_3')
cluster_labels_3.head(5)


## Build dictionaries to easily move across levels in the taxonomy based on cluster membership
esco_first_to_second={}
esco_second_to_first={}
for name,g in esco_clusters.groupby('level_1').level_2:#.value_counts():
    level_2_all = sorted(g.value_counts().index.to_list())
    esco_first_to_second[name] = level_2_all
    for level_id in level_2_all:
        esco_second_to_first[level_id] = name

esco_second_to_third={}
esco_third_to_second = {}
esco_third_to_first = {}
for name,g in esco_clusters.groupby('level_2').level_3:#.value_counts():
    level_3_all = sorted(g.value_counts().index.to_list())
    esco_second_to_third[name] = level_3_all
    for level_id in level_3_all:
        esco_third_to_second[level_id] = name
        esco_third_to_first[level_id] = esco_second_to_first[name]

esco_first_to_third = {}
for name in esco_first_to_second.keys():
    level_2_all = esco_first_to_second[name]
    level_3_all = []
    for level_id in level_2_all:
        level_3_all.append(esco_second_to_third[level_id])
    esco_first_to_third[name] = sorted(flatten_lol(level_3_all))

esco_first_to_second_label={}
esco_second_to_first_label={}
for name,g in esco_clusters.groupby('label_level_1').label_level_2:#.value_counts():
    level_2_all = sorted(g.value_counts().index.to_list())
    esco_first_to_second_label[name] = level_2_all
    for level_id in level_2_all:
        esco_second_to_first_label[level_id] = name

esco_second_to_third_label={}
esco_third_to_second_label = {}
esco_third_to_first_label = {}
for name,g in esco_clusters.groupby('label_level_2').label_level_3:#.value_counts():
    level_3_all = sorted(g.value_counts().index.to_list())
    esco_second_to_third_label[name] = level_3_all
    for level_id in level_3_all:
        esco_third_to_second_label[level_id] = name
        esco_third_to_first_label[level_id] = esco_second_to_first_label[name]

esco_first_to_third_label = {}
for name in esco_first_to_second_label.keys():
    level_2_all = esco_first_to_second_label[name]
    level_3_all = []
    for level_id in level_2_all:
        level_3_all.append(esco_second_to_third_label[level_id])
    esco_first_to_third_label[name] = sorted(flatten_lol(level_3_all))


## Load dictionary that says which ESCO-based cluster each unique skill from job advert was matched to
### Load the results of matching TK skills to ESCO skills
RESULTS_PATH = '/path/to/results'
validated_matches_file = 'tk_skills_to_skills_and_clusters_validated_final_July2020.csv'
final_matches_file = 'tk_skills_to_clusters_1to1_final_July2020.csv'

tk_esco_121 = pd.read_csv(f"{RESULTS_PATH}/{final_matches_file}", encoding = 'utf-8')
tk_esco_121 = tk_esco_121.set_index('skill_label')

# adjustments to the labels
tk_esco_121.loc[tk_esco_121.cluster_level_2==20.0,'cluster_label_level_2'
               ] = 'land transport (rail)'
tk_esco_121.loc[tk_esco_121.cluster_level_3==191.0,'cluster_label_level_3'
               ] = 'leather production (manufacturing)'
tk_esco_121.loc[tk_esco_121.cluster_level_3==190.0,'cluster_label_level_3'
               ] = 'footwear design'
tk_esco_121.loc[tk_esco_121.cluster_level_3==57.0,'cluster_label_level_3'
               ] = 'marketing (branding)'

# turn dataframe into a dictionary: access takes less time
tk_esco_121_dict = dict(zip(tk_esco_121.index,tk_esco_121.cluster_level_3))

def skill_to_cluster_list(x, tk_esco_121_dict = tk_esco_121_dict):
    ''' Transforms a list of TK skills into a list of skill clusters they are matched to'''
    clusters = [tk_esco_121_dict[t] for t in x]
    clusters = [t for t in clusters if not np.isnan(t)]
    return clusters

## Load embeddings for TK and ESCO skills
GENERATE_EMB = False
if GENERATE_EMB:
    # Generate sentence-level embeddings for each taxonomy skill
    emb_df = generate_embeddings(esco_clusters.preferred_label.to_list())

    taxonomy_2_0.tax_embeddings = pd.DataFrame(emb_df, index =
                                               esco_clusters.preferred_label.to_list())

    taxonomy_2_0.tax_embeddings.to_csv(
        f"{DATA_PATH}/interim/embeddings_esco_preferred_labels_full.gz", encoding = 'utf-8',
        compression='gzip')
else:
    loaded_df = pd.read_csv(f"{DATA_PATH}/interim/embeddings_esco_preferred_labels_full.gz",
                                              encoding = 'utf-8', compression = 'gzip')
    loaded_df = loaded_df.set_index('Unnamed: 0')
    # select relevant subset
    loaded_df = loaded_df.loc[taxonomy_2_0.skill_list]
    assert((loaded_df.index == taxonomy_2_0.skill_list).mean()==1.0)
    taxonomy_2_0.tax_embeddings = loaded_df#.to_numpy()

#%%
# Generate sentence-level embeddings for each textkernel skill
tk_skills = pd.read_csv(f"{DATA_PATH}/interim/all_types_skills_counts_batch1.csv",
                                    dtype = {'skill_label': 'string', 'counts': 'float32',
                                             'skill_type': 'category'})
if GENERATE_EMB:
    emb_df = generate_embeddings(tk_skills.skill_label.to_list())
    tk_skills_embeddings = pd.DataFrame(emb_df, index= tk_skills.skill_label)
    tk_skills_embeddings.to_csv(
                f"{DATA_PATH}/interim/embeddings_tk_skills_full.gz",
            encoding = 'utf-8',
            compression='gzip')
else:
    loaded_df = pd.read_csv(f"{DATA_PATH}/interim/embeddings_tk_skills_full.gz",
                            encoding = 'utf-8',
                            compression='gzip')
    loaded_df = loaded_df.set_index('Unnamed: 0')
    # only keep the skills of interest
    try:
        assert((loaded_df.index == tk_skills.skill_label).mean()==1.0)
    except:
        loaded_df = loaded_df.loc[tk_skills.skill_label]
        assert((loaded_df.index == tk_skills.skill_label).mean()==1.0)
    tk_skills_embeddings = loaded_df
print('Done')

comparison_vectors_bert = generate_cluster_embeddings(taxonomy_2_0.skill_list,
                                taxonomy_2_0.tax_embeddings.loc[taxonomy_2_0.skill_list],
                                bottom_layer = taxonomy_2_0.bottom_layer)

# compute weighted cluster vectors
comparison_vectors_weighted = generate_cluster_embeddings(taxonomy_2_0.skill_list,
                taxonomy_2_0.tax_embeddings.loc[taxonomy_2_0.skill_list],
                bottom_layer = taxonomy_2_0.bottom_layer,
                weights = taxonomy_2_0.skills_taxonomy_full.coreness_at_level_3.to_numpy())


''' Compute most representative skill cluster for each job advert'''

# Process one chunck at a time using the following stesps:
#
# 1. Join up all the skills into a skill list per job adverts
# 2. Link skills to clusters
# 3. Find most representative cluster for each job adverts based on weighted average similarities

# get filenames for the data
N_to_load = tk_params.N_files
indices_to_load = range(tk_params.N_files)
dfilenames = [os.path.join(f"{DATA_PATH}/processed",tk_params.file_name_template.format(i))
              for i in indices_to_load]
import_dict, dates_to_parse = create_tk_import_dict()


# Create "discount vectors for level 3 clusters"
counts3 = tk_esco_121.groupby('cluster_level_3').tk_counts.sum()
logcounts3 = -np.log(counts3/counts3.sum())
logcounts3 = (logcounts3 - min(logcounts3)) / (max(logcounts3) - min(logcounts3))
logcounts3 = 0.4*logcounts3 + 0.8
for i in np.arange(201.0):
    try:
        logcounts3.loc[i]
    except:
        logcounts3.loc[i] = 1

# pre-compute all possible similarities
all_cosine_similarities = cosine_similarity(tk_skills_embeddings.iloc[:100000].to_numpy(),
                                            comparison_vectors_bert)
full_similarities_df = pd.DataFrame(all_cosine_similarities,
                                    index = tk_skills_embeddings.index,
                                   columns = comparison_vectors_bert.index)

# multiply by the "discount" factors
for col in full_similarities_df.columns:
    full_similarities_df.loc[:,col] = full_similarities_df.loc[:,col]*logcounts3.loc[col]


# get average vectors for second level clusters and the auto-similarity matrix
comparison_vectors_2 = deepcopy(comparison_vectors_bert)
comparison_vectors_2['level2'] = comparison_vectors_2.index.map(
    lambda x: esco_third_to_second[x])
comparison_vectors_2 = comparison_vectors_2.groupby('level2').mean()

#
self_similarity_level2 = pd.DataFrame(cosine_similarity(comparison_vectors_2),
                                      index= comparison_vectors_2.index,
                                      columns = comparison_vectors_2.index)


# ## Define functions that depend on similarity vectors
def compute_top_cluster(df_row):
    ''' Compute most representative ESCO cluster given a list of TK skills'''
    if len(df_row.cluster_unique):
        sims = full_similarities_df.loc[df_row.skill_vector,df_row.cluster_unique].to_numpy()
        weights = self_similarity_level2.loc[df_row.cluster_unique,
                                             df_row.cluster_unique].to_numpy()
        discounted_sims = (sims.mean(axis=0))*(weights.mean(axis=0))
        return df_row.cluster_unique[discounted_sims.argmax()]
    else:
        return pd.NA


# Here is to actually compute the most representative cluster
cols_to_load = ['textkernel/columns/to/load']

out_timer = TaskTimer()
out_timer.start_task()
filename_base = f"{DATA_PATH}/processed/" + tk_params.file_name_template

# process one chunck of data at a time
for i in tqdm(range(f_start,f_end)):
    file_name = filename_base.format(i)
    data_df = pd.read_csv(file_name, compression='gzip',encoding = 'utf-8', usecols = cols_to_load)
    # remove adverts not in english and extra column
    data_df = data_df.loc[data_df.flags.map(lambda x: x%100 != 11)]

    # add joined up skill vector column
    data_df = data_df.assign(skill_vector = join_skill_columns(data_df, SOFT= False))
    data_df.skill_vector = data_df.skill_vector.astype('str')

    # process skill vector column and link each skill to ESCO clusters
    data_df.skill_vector = data_df.skill_vector.map(lambda x: split_skill_vector(x))
    data_df['cluster_vector'] = data_df.skill_vector.map(lambda x: skill_to_cluster_list(x))

    data_df['cluster_unique'] = data_df.cluster_vector.map(lambda x: list(set(x)))
    data_df['cluster_unique_2'] = data_df.cluster_unique.map(lambda x:
                                                [esco_third_to_second[t] for t in x])


    best_clusters = pd.DataFrame(index = data_df.posting_id,
                             columns = ['top_cluster','top_cluster_weighted'])

    # process each row - haven't figured out a faster way
    for _,df_row in data_df.iterrows():
        name = df_row.posting_id
        num_clusters= len(df_row.cluster_unique)
        if num_clusters==0:
            continue
        elif num_clusters == 1:
            best_clusters.loc[name,'top_cluster'] = df_row.cluster_unique[0]
            best_clusters.loc[name,'top_cluster_weighted'] = df_row.cluster_unique[0]
        else:
            sims = full_similarities_df.loc[df_row.skill_vector,df_row.cluster_unique
                                           ].to_numpy()

            weights = self_similarity_level2.loc[df_row.cluster_unique_2,
                                                 df_row.cluster_unique_2].to_numpy()

            discounted_sims = sims.mean(axis=0)
            best_clusters.loc[name,'top_cluster'] = df_row.cluster_unique[
                    discounted_sims.argmax()]
            # now with weights (context factor)
            discounted_sims *= weights.mean(axis=0)

            best_cluster = df_row.cluster_unique[discounted_sims.argmax()]
            best_clusters.loc[name,'top_cluster_weighted'] = best_cluster

    best_clusters = best_clusters.assign(profession_soc_code_value = data_df['profession_soc_code_value'].values)

    best_clusters.to_csv(f"{DATA_PATH}/interim/filename{i}.gz",
        encoding='utf-8', compression = 'gzip')
