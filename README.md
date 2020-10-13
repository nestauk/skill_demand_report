# skill_demand_report
Last updated: 14th October 2020.

This Github repo is to hold supplementary material for Nesta's report on "Developing experimental estimates of skill demand", produced as part of a research project funded by the Economics Statistics Centre of Excellence (ESCoE).

# Authors
Stef Garasto, Jyldyz Djumalieva, Karlis Kanders, Rachel Wilcock and Cath Sleeman.

# Content
This repository contains three folders.

One ('report_pdf') has a pdf copy of the report. This is a copy of the version submitted for the ESCoE discussion paper, but it has not been reviewed yet. If needed, this repository will be updated with a link to (or a copy of) the published version [ADD LINK].

The other two folders contain supplementary tables with the stock of vacancies and samples of the code used for the report. Because of legal restrictions, it is not possible to add any of the underlying online job adverts data. No data can be published.

## Tables
In the folder 'tables_report' there are files storing the annual stock of vacancies covering a period of 5 years, from 2015 to 2019. The stock of vacancies is broken down by one variable per file (that is, only by location or only by industry). All the variables used for the breakdown are:
<ul>
<li>Occupations (at all levels of granularity of the Standard Occupational Classification, SOC2010).</li>
<li>Industry (broad categories from the Standard Industrial Classification, SIC 2007).</li>
<li>Location (Travel To Work Areas).</li>
<li>Skill categories (at all levels of Nesta's skills taxonomy). </li>
</ul>

The stock of vacancies by Travel To Work Areas (TTWAs) is given as the stock of vacancies normalised by 100 economically active residents (source: Annual Population Survey) and estimates are shown only for TTWAs with at least 40,000 economically active residents aged 16 and over. For all the other variables the estimates are expressed as percentages (they sum up to 100 for each year).

Each file is provided in two formats:
<ul>
<li> '.csv' - machine readable format. The first column contains the value of the breakdown variable, then there is one column per year.</li>
<li> '.xlsx' - less machine readable, but contains more human readable information on the data provided.</li>
</ul>

The dataset of online job adverts on which these estimates are based was provided by Textkernel.

## Sample code
In the folder 'sample_code' there are pieces of code underlying key elements of the methods used to produced the estimates of skill demand (more details can be found in the report). Specifically, we provide code showing how we:
<ul>
<li>Built a crosswalk from SOC2010 to SIC2007 (<code>01-SIC-REW.ipynb</code>).</li>
<li>Computed the most representative skill cluster for each job advert (<code>compute_top_clusters.py</code>).</li>
<li>Converted the flow of job adverts into a stock (<code>flow_to_stock_funcs.py</code> and <code>flow_to_stock_model_by_sic_averaged_Sep20.ipynb</code>).</li>
<li>Aligned the stock of online job adverts to the stock of ONS vacancies from the ONS Vacancy survey (<code>flow_to_stock_model_by_sic_averaged_Sep20.ipynb</code>).</li>
</ul>

The code is provided "as is".
