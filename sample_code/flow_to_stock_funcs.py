"""
Created by Stef Garasto.
Last updated: September 2020

Description:
This file contains the main functions to turn the flow of job adverts into
a stock. For a more detailed explanation of the models, see
a) the ESCoE report on "Developing experimental estimates of regional skill demand".
b) This Google doc: ADD LINK.

There is also a function needed for the reweighting procedure to align stock of
online job adverts to stock of ONS vacancies.
"""

# Imports
import pandas as pd
try:
    from utils_general import printdf
except:
    import sys
    sys.path.append("/Users/stefgarasto/Local-Data/scripts/skill_demand_escoe/skill_demand/skill_demand")
    from utils.utils_general import printdf

# helpers
def set_month_at_beginning(x):
    """Set a datetime to the beginning of the month"""
    return pd.offsets.MonthBegin(0).rollback(x)

def set_month_at_end(x):
    """ Set a datetime to the end of the month"""
    return pd.offsets.MonthEnd(0).rollforward(x)

#%% -------------------------------------------------------------------------
#              Main functions to convert from flow to stock
#% --------------------------------------------------------------------------


def get_stock(data, agg_func = 'sum', agg_col = 'vacancy_weight', BOUNDARY = None):
    """
    Compute the daily stock of vacancies via cumulative sum of net Flow.
    (Turrell, A., Speigner, B., Djumalieva, J., Copple, D. and Thurgood, J. (2018)
    Using job vacancies to understand the effects of labour market mismatch on UK
    output and productivity.).
    It only computes total stock, without any breakdown.

    Keyword arguments:
    data -- dataframe with online job vacancies. Need to have "date", "end_date" and agg_col columns
    agg_func: whether to count the vacancies or to sum the weights
    agg_col -- reference column to aggregate (usually column with per-vacancy weights)
    BOUNDARY -- what to do wrt boundary conditions (start and end month)

    Note: the part of the code related to the boundary options do not really work
    as they are. The only accepted option at the moment is to use BOUNDARY = None
    and then discard the first two months of stock (empirically, I might even
    suggest to discard the first three).
    """

    start_day = data.date.min()
    end_day = data.date.max()

    if agg_func == 'sum':
        vacancy_flow_per_day = data.groupby('date')[agg_col].sum()
        vacancy_remove_per_day = data.groupby('end_date')[agg_col].sum()
    else:
        vacancy_flow_per_day = data.groupby('date')[agg_col].count()
        vacancy_remove_per_day = data.groupby('end_date')[agg_col].count()

    # shift vacancy_remove_per_day by one day since vacancies disappear the day after their expiration date
    vacancy_remove_per_day = vacancy_remove_per_day.shift(1)

    # adjust so that they start and end on the same dates
    vacancy_flow_per_day = vacancy_flow_per_day.reindex(pd.date_range(start=start_day,
                                        end=end_day,freq='D'), fill_value =0)
    vacancy_remove_per_day = vacancy_remove_per_day.reindex(pd.date_range(start=start_day,
                                        end=end_day,freq='D'), fill_value =0)

    # compute the net Flow
    net_flow = vacancy_flow_per_day.fillna(0) - vacancy_remove_per_day.fillna(0)

    # Get the daily stock
    daily_stock = net_flow.cumsum()

    # Resample to monthly stock
    monthly_stock = net_flow.resample('M').sum().cumsum()
    monthly_stock.index = monthly_stock.index.map(set_month_at_beginning)

    # enforce boundary conditions
    if BOUNDARY == 'valid':
        monthly_stock = monthly_stock[monthly_stock.index>=set_month_at_beginning(
            pd.to_datetime(FIRST_VALID_MONTH))]
    return monthly_stock, daily_stock, vacancy_flow_per_day, vacancy_remove_per_day


def get_stock_breakdown(data, agg_func = 'sum', agg_col = 'vacancy_weight',
                              breakdown_col = 'organization_industry_value', BOUNDARY = None):
    """
    Compute the daily stock of vacancies via cumulative sum of net Flow.
    (Turrell, A., Speigner, B., Djumalieva, J., Copple, D. and Thurgood, J. (2018)
    Using job vacancies to understand the effects of labour market mismatch on UK
    output and productivity.).
    It computes the stock broken down by other variables of interest, like the
    industry code.

    Keyword arguments:
    data -- dataframe with online job vacancies. Need to have "date", "end_date" and agg_col columns
    agg_func: whether to count the vacancies or to sum the weights
    agg_col -- reference column to aggregate (usually column with per-vacancy weights)
    BOUNDARY -- what to do wrt boundary conditions (start and end month)

    Note: the part of the code related to the boundary options do not really work
    as they are. The only accepted option at the moment is to use BOUNDARY = None
    and then discard the first two months of stock (empirically, I might even
    suggest to discard the first three).

    Note 2: There is a division by 2 based on the fact that the ONS vacancy survey
    is only open for two weeks per year. If we adjust the stock on online vacancies
    by that of ONS vacancies, this factor of 2 only affects the magnitude of the
    adjustment weights and NOT the final stock results. However, it does make it
    look like there are less online job adverts vacancies that there actually are.
    Would suggest removing it in future - it's being kept so far for consistency
    with the ESCoE report on skill demand.
    """

    start_day = data.date.min()
    end_day = data.date.max()

    if agg_func == 'sum':
        vacancy_flow_per_day = data.groupby(['date',breakdown_col])[agg_col].sum()
        vacancy_remove_per_day = data.groupby(['end_date',breakdown_col])[agg_col].sum()
    else:
        vacancy_flow_per_day = data.groupby(['date',breakdown_col])[agg_col].count()
        vacancy_remove_per_day = data.groupby(['end_date',breakdown_col])[agg_col].count()

    vacancy_flow_per_day = vacancy_flow_per_day.unstack()
    vacancy_remove_per_day = vacancy_remove_per_day.unstack()

    # shift vacancy_remove_per_day by one day since vacancies disappear the day after their expiration date
    vacancy_remove_per_day = vacancy_remove_per_day.shift(1)

    # adjust so that they start and end on the same dates
    vacancy_flow_per_day = vacancy_flow_per_day.reindex(pd.date_range(start=start_day,
                                        end=end_day,freq='D'), fill_value =0)
    vacancy_remove_per_day = vacancy_remove_per_day.reindex(pd.date_range(start=start_day,
                                        end=end_day,freq='D'), fill_value =0)

    # compute the net Flow
    net_flow = vacancy_flow_per_day.fillna(0) - vacancy_remove_per_day.fillna(0)

    # Get the daily stock
    daily_stock = net_flow.cumsum()

    # Resample to monthly stock
    monthly_stock = net_flow.resample('M').sum().cumsum()/2
    monthly_stock.index = monthly_stock.index.map(set_month_at_beginning)

    # enforce boundary conditions
    if BOUNDARY == 'valid':
        monthly_stock = monthly_stock[monthly_stock.index>=set_month_at_beginning(
            pd.to_datetime(FIRST_VALID_MONTH))]
    return monthly_stock, daily_stock, vacancy_flow_per_day, vacancy_remove_per_day

## SLIGHTLY DIFFERENT FLOW TO STOCK MODEL
#%%
def get_stock_v0(data, agg_func = 'count', agg_col = 'vacancy_weight', BOUNDARY = None):
                    #start_day = '2015-03-01', end_day = '2019-10-31',
#                    GET_END_DATE = False, BOUNDARY = None):
    """
    Compute the daily stock of vacancies via difference of cumulative sums (
    cumulative sum of daily inbound flow - cumulative sum of daily outbound flow
    ). The resulting daily stock is then resampled to monthly.

    The advantage of using this implementation is that we can make better use of
    vacancies that are not added and removed in the same month, BUT it is much
    more sensitive to initial conditions, so it will overestimate the stock
    (because at the beginning vacancies are only added and not deleted).
    Ultimately though, this does not matter much IF we REWEIGHT the stock based
    on official vacancy numbers. This is because the reweighting procedure will
    make the stock level look similar to the official statistics, irrespective
    of where we start from.

    Suggestions for future improvement: implement a boundary condition that allows
    us to do this:
    df_stock = df_start[effective_start_date:].cumsum(
                                        ) - df_end[effective_start_date:].cumsum()

    Keyword arguments:
    data -- dataframe with online job vacancies. Need to have "date", "end_date" and agg_col columns
    agg_func: whether to count the vacancies or to sum the weights
    agg_col -- reference column to aggregate (usually column with per-vacancy weights)
    BOUNDARY -- what to do wrt boundary conditions (start and end month)
    """

    '''
    # obs: taking the difference of the cumulative sum is the same as doing this:
    df_stock2 = pd.DataFrame(columns = ['open_vacancies'],
                             index= pd.date_range(start='2015-03-13',end='2019-10-31',freq='D'))
    df_stock2.open_vacancies = 0
    for reference_day in tqdm(df_stock.index):
        df_stock2.loc[reference_day] = ((data.date<=reference_day) &
                               (data.end_date>reference_day)).sum()

    '''

    start_day = data.date.min()
    end_day = data.date.max()

    if agg_func == 'count':
        df_start = data.groupby('date').count()[agg_col].to_frame()#.reset_index()
        df_end = data.groupby('end_date').count()[agg_col].to_frame()#.reset_index()
    elif agg_func == 'sum':
        df_start = data.groupby('date').sum()[agg_col].to_frame()#.reset_index()
        df_end = data.groupby('end_date').sum()[agg_col].to_frame()#.reset_index()
    else:
        print('Wrong aggregate function')
        assert(agg_func in ['count','sum'])

    #df_start = df_start.set_index('date').resample('1D').mean().fillna(0)
    #df_end = df_end.set_index('end_date').resample('1D').mean().fillna(0)
    # shift the end date by one, since vacancies disappear the day after their expiration date
    df_end = df_end.shift(1)

    df_start = df_start.reindex(pd.date_range(start=start_day,
                                    end=end_day,freq='D'), fill_value =0)
    df_end = df_end.reindex(pd.date_range(start=start_day,
                                    end=end_day,freq='D'), fill_value =0)

    # compute daily stock
    df_stock = df_start.cumsum() - df_end.cumsum()

    # add boundary conditions, if requested
    if BOUNDARY == 'valid':
        #TODO: review
        # start from the month after the first date
        if not df_stock.index[0].is_month_start:
            valid_start = pd.offsets.MonthBegin(0).rollforward(df_stock.index[0])
        else:
            valid_start = pd.offsets.MonthBegin(0).rollforward(df_stock.index[1])
        print(f'Starting from {valid_start}')
        df_stock = df_stock[df_stock.index>=valid_start]

    # resample to monthly
    stock_month1 = df_stock.resample('M').mean()
    stock_month1.index = stock_month1.index.map(set_month_at_beginning)

    # Remove first month if we want to avoid the boundary
    if BOUNDARY=='valid':
        stock_month1 = stock_month1[stock_month1.index>=set_month_at_beginning(
            pd.to_datetime(FIRST_VALID_MONTH))]
    return stock_month1, df_stock, df_start, df_end

def load_ons_vacancies(DATA_PATH, start_date = '2015-03-01', end_date = '2019-10-31'):
    """
    Load and process ONS monthly, non-seasonally adjusted vacancy data [dataset x06].
    This function is tailored to the current grouping of some SIC codes.

    Note: it assumes the dataset has been downloaded and put in the data/aux folder path.

    Arguments:
    start_date = first day of month from when to start the time series
    end_date = last day of month from when to end the time series
    """
    ons_df = pd.read_excel(f"{DATA_PATH}/aux/x06apr20.xls", sheet_name='Vacancies by industry',
                          skiprows = 3)
    # keep columns with data and clean the column names
    ons_df = ons_df[[col for col in ons_df.columns if not 'Unnamed:' in col]]
    cleaned_col_names = {}
    for col in ons_df.columns[2:]:
        cleaned_col = col.replace('&', 'and').replace('-','').replace(',','').lower()
        cleaned_col = ''.join([t for t in cleaned_col if not t.isdigit()])
        cleaned_col_names[col] = '_'.join(cleaned_col.split())
    # manual adjustment for one column
    cleaned_col_names['Manu-    facturing'] = 'manufacturing'
    cleaned_col_names['SIC 2007 sections'] = 'month'
    cleaned_col_names['All vacancies1 '] = 'vacancies'
    ons_df = ons_df.rename(columns = cleaned_col_names)
    # extract the row with the letters
    sic_letters = ons_df.iloc[0]
    # remove empty rows
    ons_df = ons_df.loc[(ons_df.month.notna()) & (ons_df.vacancies.notna())]
    # join up some industries
    ons_df = ons_df.assign(wholesale_retail_motor_trade_and_repair =
                           ons_df.motor_trades + ons_df.wholesale + ons_df.retail)
    ons_df = ons_df.assign(wholesale_and_retail = ons_df.wholesale + ons_df.retail)
    #ons_df = ons_df.assign(others = ons_df.vacancies - ons_df[partial_map_tk2sic.values()].sum(axis=1))

    ons_df = ons_df.assign(education_and_professional_activities = ons_df.education +
                                                        ons_df.professional_scientific_and_technical_activities)
    ons_df = ons_df.assign(utilities = ons_df.electricity_gas_steam_and_air_conditioning_supply +
                           ons_df.water_supply_sewerage_waste_and_remediation_activities)
    ons_df = ons_df.assign(personal_and_public_services = ons_df.real_estate_activities +
                                                    ons_df['public_admin_and_defence;_compulsory_social_security'] +
                                                    ons_df.other_service_activities)
    sic_letters.loc['wholesale_retail_motor_trade_and_repair'] = 'G'
    sic_letters.loc['wholesale_and_retail'] = 'G46_47'
    sic_letters.loc['education_and_professional_activities'] = 'M_P'
    sic_letters.loc['utilities'] = 'D_E'
    sic_letters.loc['personal_and_public_services'] = 'L_O_S'
    sic_letters.loc['others'] = 'Z'
    sic_letters.loc['vacancies'] = 'vacancies'
    #
    ons_df.month = pd.to_datetime(ons_df.month)
    # only need vacancies within a certain period
    ons_df = ons_df[(ons_df.month>=pd.to_datetime(start_date)) &
                    (ons_df.month<=pd.to_datetime(end_date))]
    ons_df = ons_df.set_index('month')
    return ons_df, sic_letters


def scale_weights_by_total_levels(weights_df, ons_vacancies, raw_stock_per_month_sic,
                            sectors_in_common = ['B', 'C', 'D_E', 'F', 'G', 'H', 'I',
                                    'J', 'K', 'L_O_S', 'M_P', 'N', 'Q', 'R']):
    """
    This function is used to scale the per-vacancy weights used to align the total
    stock of online job vacancies with the total stock of vacancies from the ONS survey.

    Arguments:
    weights_df = dataframe of adjustment weights for each month and sector. Has
                3 columns: 'best_month', 'final_sic_letter' and 'vacancy_weight'
    ons_vacancies = stock of vacancies from ONS survey (index = month, columns = sectors)
    raw_stock_per_month_sic = un-weighted stock of vacancies from online job adverts
                    (index = month, columns = sectors)
    sectors_in_common = list of sectors for which we can compute a ratio
                        ONS_stock/online_job_adverts_stock

    RATIONALE
    This is needed because there are some job adverts that are classified as uncertain
    and the per-vacancy weights are computed without including this uncertain posts.
    The weights are computed as a ratio between the two stocks broken down by SIC
    and month (ONS/online adverts), and then uncertain job adverts are assigned the
    median weight for that month. Overall this results into a total stock count
    that is now higher than the ONS one because the overall sum includes vacancies
    that were not used to compute the weights. However, if I include them then the
    absolute levels by SIC from OJA will be artificially smaller than the absolute levels
    from the ONS because the weights have been decreased by including the uncertain vacancies.

    It is a necessary trade-off whose importance will diminish the fewer job adverts
    there are that we can't reliably assign to a SIC code.

    In the end, this second re-weighting step was added because arguably those uncertain
    jobs do have a SIC code, it's just that we don't know which one it is yet.
    Because of this, it was thought that it would be best to fully align the total
    stock levels. This means that the total stock level when analysing breakdowns
    by other variables would not be artificially inflated.

    METHOD
    Multiply each weight by an extra factor that depends only on the month.
    This factor is given by:

    w_month = 1/(alpha + 1)

    with:
    alpha = x_j/z * w_bar
    z = sum_i {z_i}
    w_bar = weight for uncertain job adverts (average or median across other sectors)
    (if the average w_bar = 1/N * sum_i {z_i/x_i})

    x_j = raw online vacancy stock for "uncertain" adverts in a given month.
            raw means not re-weighted.
    x_i = raw online vacancy stock for sector i in a given month
    z_i = ONS vacancy stock for sector i in a given month
    z_i / x_i weight for sector i in a given month
    N = # of sectors considered (these are only the sectors present in both online
        job adverts data and in the ONS vacancy data)

    Note that the original formula was derived in the case of the weight for uncertain
    job adverts being computed as the mean across other sectors. So far, we have
    actually used the median because of the outliers in the mining sector. However,
    a) we checked and the median is very close to the mean computed after removing
    the mining sector, so the method still provides a good first approximation.
    b) the method might still be valid with the median as well.
    """
    #all_sic_codes = df_data.final_sic_letter.value_counts().index.tolist()
    #columns_unlike_ons = [t for t in all_sic_codes if t not in sectors_in_common]
    all_weights2 = weights_df.pivot(index='best_month',columns = 'final_sic_letter', values = 'vacancy_weight')
    #all_weights['vacancy_weight'].unstack(level=1)#reset_index(level=0)
    #tmp = {}
    w_bar = {}
    alpha = {}
    gamma = {}
    ons_sum = {}
    extra_stock = {}
    # compute adjustment month by month
    for month in all_weights2.index:
        #tmp[month] = all_weights2.loc[month][sectors_in_common].sum()#.loc[sectors_in_common])
        w_bar[month] = all_weights2.loc[month]['uncertain']
        # ONS stock is expressed as (thousands of vacancies)
        ons_sum[month] = 1000.0*ons_vacancies.loc[month,sectors_in_common].sum()
        extra_stock[month] = raw_stock_per_month_sic.loc[month,'uncertain'].sum()
        alpha[month] = extra_stock[month]/ons_sum[month]*w_bar[month]
        gamma[month] = 1/(1+alpha[month])
        #print(ons_sum[month],extra_stock[month],w_avg[month],alpha[month],gamma[month])

    printdf(pd.DataFrame.from_dict(gamma, orient = 'index', columns = ['vacancy_weight_gamma']).head())
    all_weights3 = all_weights2.reset_index().melt(id_vars = 'best_month').merge(
                        pd.DataFrame.from_dict(gamma, orient = 'index', columns = ['vacancy_weight_gamma']),
                                    left_on = ['best_month'], right_index= True, how='left')
    all_weights3 = all_weights3.assign(vacancy_weight_new = all_weights3.value * all_weights3.vacancy_weight_gamma)
    return all_weights3
