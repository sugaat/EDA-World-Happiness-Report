#!/usr/bin/env python
# coding: utf-8

# # EDA On World Happiness Report 2023😀
# 
# 
# ### INTRODUCTION
# The World Happiness Report is a yearly study that looks at how happy people are around the world. It ranks countries based on things like how rich they are, how much support people have, and how long they live. The report helps governments understand what makes people happy and how they can make life better for their citizens.
# 
# The chosen dataset pertains to the 2023 World Happiness Report, available on Kaggle. This dataset ranks countries globally based on diverse factors contributing to happiness, encompassing economic indicators, social support, life expectancy, freedom to make life choices, generosity, and perceptions of corruption. The summation of these factors yields the happiness. Higher the happiness score, the higher the happiness rank.The interpretation of these factors lies in their impact on happiness levels across nations.
# 
#  
# By analyzing the World Happiness Report and providing actionable insights, valuable contributions are made to the broader discussion on well-being. These efforts aid stakeholders in making data-driven decisions to enhance the quality of life in different regions and countries.
# 
# 
# <p><img src="Smiley.jpg" width="200"><img src="imgs/happy-cartoon-gif.gif" width="200"></p>
# 
# ### Objective:
# The purpose of choosing this work aims to identify the key factors that contribute significantly to leading a more joyful life. Consequently, individuals and nations can prioritize these pivotal factors to attain elevated levels of happiness.
#  

# ### Citation(Source):
# * https://worldhappiness.report/ed/2023/
# * https://www.kaggle.com/datasets/andradaolteanu/country-mapping-iso-continent-region
# * https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2023
# 
# ![a](imgs/image_processing.gif)
# 

# The following features were measured and examined:
# 1) Ladder score (Happiness score)	
# 2) Logged GDP per capita	
# 3) Social support	
# 4) Healthy life expectancy	
# 5) Freedom to make life choices
# 6) Generosity	
# 7) Perceptions of corruption	
# 8) Region

# ## Snapshot of Datasets
# 
# |    HAPPINESS REPORT    |   CONTINENTS MAPPING   |
# | ---------------------- | ---------------------- |
# | ![cat](whr_ss.png)     | ![dog](continents_ss.png) |
# 

# 
# ### Importing Libraries and Reading the data file

# In[1]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px

plt.style.use('ggplot')


import plotly.io as pio
pio.templates.default = "plotly_white"


# In[2]:


# Reading the data files
WHR = pd.read_csv('WHR2023.csv')      # WHR_2023 is a dataframe
country_mapping = pd.read_csv('continents2.csv')     #country_mapping is a data frame


# ## Understanding the data
# **WHR**

# In[3]:


# quick look to data set
WHR.head()


# In[46]:


WHR.tail()


# In[4]:


# How many rows and columns in the data set
WHR.shape


# 
# * There are 19 columns (feature)
# * There are 137 data (tuples) 

# In[5]:


# Prints the columns
WHR.columns


# In[6]:


# Prints summary statistics
WHR.describe()


# * Max ladder Score is 7.8
# * Minimum Score is 1.8
# * Average Score is 5.5
# 

# In[7]:


# data types of the World Happiness 2023 dataset column
WHR.dtypes


# In[8]:


# Complete information about the data set
WHR.info()


# **Country mapping** 

# In[9]:


country_mapping.head()


# In[10]:


country_mapping.shape


# In[11]:


country_mapping.columns


# In[12]:


country_mapping.dtypes


# In[13]:


country_mapping.info()


# ### Preparing the data

# In[14]:


WHR 


# In[15]:


WHR.drop(WHR.loc[:, 'Standard error of ladder score':'lowerwhisker'].columns, inplace=True, axis=1)

WHR.drop(WHR.loc[:, 'Ladder score in Dystopia':'Dystopia + residual'].columns, inplace=True, axis=1)
# Remove all columns between column index 1 to 3


# In[16]:


WHR


# In[17]:


WHR.isnull().sum()


# In[18]:


# Inspect missing values in the dataset
print(WHR.isnull().values.sum())

# Replace the ' 's with Nan
WHR = WHR.replace(" ", np.NaN)

# Impute the missing values with mean imputation
WHR = WHR.fillna(WHR["Healthy life expectancy"].mean())

# Count the number of Nans in the dataset to verify
print(WHR.isnull().values.sum())


# In[19]:


country_mapping.head()


# In[20]:


country_mapping = country_mapping.drop(country_mapping.columns[[1,3,4,7,8,9,10]], axis = 1)
# delete the columns 1,3,4,7,8,9,10


# In[21]:


country_mapping.head()


# In[22]:


#Rename the columns for consistency
country_mapping = country_mapping.rename({'name':'Country name',
                                          'alpha-3':'iso_alpha',
                                          'sub-region':'sub_region'}, axis =1)


# In[23]:


country_mapping


# In[24]:


#Merge the happiness data and country mapping dataframes

full_data = WHR.merge(country_mapping, on='Country name', how='left')
full_data


# In[25]:


# Check for rows with null region that did not match correctly when merging
nan_region_rows = full_data[full_data['region'].isnull()]
nan_region_rows


# In[26]:


full_data.dropna(subset=['region'], inplace=True)



# In[27]:


#Check for nulls
full_data.isnull().sum()


# In[45]:


full_data.describe()


# ## Visualization:

# In[28]:


#  the top 5 happiest countries
top5 = WHR[['Country name', 'Ladder score']].head(5)
print(top5)


# In[29]:


#  the least 5 happiest countries
bottom5 = WHR[['Country name', 'Ladder score']].tail(5)
print(bottom5)


# In[30]:


plt.subplot(2,1,1)
sns.barplot(top5, x=top5['Ladder score'], y=top5['Country name'] ,color = 'green', edgecolor='white')

plt.subplot(2,1, 2)
sns.barplot(bottom5, x=bottom5['Ladder score'], y=bottom5['Country name'] ,color = 'red', edgecolor='white')


# In[31]:


plt.hist(WHR['Ladder score'], bins = 8 , edgecolor = 'white')

plt.title('Happiness Score Distribution')
plt.xlabel('Score')
plt.ylabel('No of countries')


# INTERPRETATION:
# * We can see that most of the countries are concenrated near to score 6
# * There are least amount of countries with ladder score between 1 and 3

# In[32]:


full_data.head(2)


# In[33]:


plt.figure(figsize=[11, 7])

# scatter plot with regression line for 'Logged GDP per capita'
plt.subplot(2,3,1)
sns.scatterplot(data = full_data, x='Ladder score', y= 'Logged GDP per capita', hue = 'region')
sns.regplot(data = full_data, x='Ladder score', y= 'Logged GDP per capita', scatter = False, ci=None) # ci is the blue bg
plt.xlabel('Happiness', fontsize=8)
plt.ylabel('Logged GDP per capita', fontsize=8)
plt.grid(True)

# scatter plot with regression line for 'Social support'
plt.subplot(2,3,2)
sns.scatterplot(data = full_data, x='Ladder score', y= 'Social support', hue = 'region')
sns.regplot(data = full_data, x='Ladder score', y= 'Social support', scatter = False, ci=None)
plt.xlabel('Happiness', fontsize=8)
plt.ylabel('Social support', fontsize=8)

# scatter plot with regression line for 'Healthy life expectancy'
plt.subplot(2,3,3)
sns.scatterplot(data = full_data, x='Ladder score', y= 'Healthy life expectancy', hue = 'region')
sns.regplot(data = full_data, x='Ladder score', y= 'Healthy life expectancy', scatter = False, ci=None)
plt.xlabel('Happiness', fontsize=8)
plt.ylabel('Healthy life expectancy', fontsize=8)

# scatter plot with regression line for 'Freedom to make life choices'
plt.subplot(2,3,4)
sns.scatterplot(data = full_data, x='Ladder score', y= 'Freedom to make life choices', hue = 'region')
sns.regplot(data = full_data, x='Ladder score', y= 'Freedom to make life choices', scatter = False, ci=None)
plt.xlabel('Happiness', fontsize=8)
plt.ylabel('Freedom to make life choices', fontsize=8)

# scatter plot with regression line for 'Generosity'
plt.subplot(2,3,5)
sns.scatterplot(data = full_data, x='Ladder score', y= 'Generosity', hue = 'region')
sns.regplot(data = full_data, x='Ladder score', y= 'Generosity', scatter = False, ci=None)
plt.xlabel('Happiness', fontsize=8)
plt.ylabel('Generosity', fontsize=8)

# scatter plot with regression line for 'Perceptions of corruption'
plt.subplot(2,3,6)
sns.scatterplot(data = full_data, x='Ladder score', y= 'Perceptions of corruption', hue = 'region')
sns.regplot(data = full_data, x='Ladder score', y= 'Perceptions of corruption', scatter = False, ci=None)
plt.xlabel('Happiness', fontsize=8)
plt.ylabel('Perceptions of corruption', fontsize=8)

plt.tight_layout()
plt.show()


# INTERPRETATION:
# * GDP, Freedom, Life expectancy, Social Support, Freedom are directly related with happiness.
# * Generosity, perceptions of corruption do not have direct relation with Happiness.
# 
# 

# In[34]:


# sns.lineplot(data = df, x='absences', y='age')

plt.figure(figsize=[15, 9])

plt.subplot(3,2,1)
sns.lineplot(data = full_data, x='Ladder score', y= 'Logged GDP per capita', hue = 'region')
plt.xlabel('Happiness', fontsize=10)
plt.ylabel('Logged GDP per capita', fontsize=10)
plt.grid(True)

plt.subplot(3,2,2)
sns.lineplot(data = full_data, x='Ladder score', y= 'Social support', hue = 'region')
plt.xlabel('Happiness', fontsize=10)
plt.ylabel('Social support', fontsize=10)

plt.subplot(3,2,3)
sns.lineplot(data = full_data, x='Ladder score', y= 'Healthy life expectancy', hue = 'region')
plt.xlabel('Happiness', fontsize=10)
plt.ylabel('Healthy life expectancy', fontsize=10)

plt.subplot(3,2,4)
sns.lineplot(data = full_data, x='Ladder score', y= 'Freedom to make life choices', hue = 'region')
plt.xlabel('Happiness', fontsize=10)
plt.ylabel('Freedom to make life choices', fontsize=10)

plt.subplot(3,2,5)
sns.lineplot(data = full_data, x='Ladder score', y= 'Generosity', hue = 'region')
plt.xlabel('Happiness', fontsize=10)
plt.ylabel('Generosity', fontsize=10)

plt.subplot(3,2,6)
sns.lineplot(data = full_data, x='Ladder score', y= 'Perceptions of corruption', hue = 'region')
plt.xlabel('Happiness', fontsize=10)
plt.ylabel('Perceptions of corruption', fontsize=10)

plt.tight_layout()
plt.show()


# In[35]:


# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10,6))

# Set title and labels for x and y axes
plt.title("Happiness score boxplot by region", fontsize = 20)
sns.boxplot(data=full_data, y="region", x="Ladder score", palette='Greens_r', order=["Oceania", "Europe", "Americas", "Asia", "Africa"] ).set(
    xlabel='Happiness Score', 
    ylabel='Region'
)

# Display the figure
plt.show()


# INTERPRETATION
# 
# * The plot suggests that there is not a significant gap between the most and least happy nations.
# 
# * Within each region, there is a mix of extremely happy and extremely unhappy individuals.
# 
# * Asia and Africa, have a more diverse range of happiness scores compared to others like Oceania, Europe, and Americas.
# 
# * The boxplot for the region "Oceania" is shorter in length compared to other regions, indicating that happiness scores in Oceania are more concentrated and less spread out.
# 
# * The boxplot for "Asia" has the largest length, indicating that happiness scores in Asia are the most spread out, with a larger range of values.
# 
# * The boxplot for "Africa" is relatively short in length, suggesting that happiness scores in Africa are also more concentrated and less spread out compared to other regions.
# 

# In[36]:


# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10,6))

# Set title and labels for x and y axes
plt.title("Happiness score boxplot by region", fontsize = 20)
ax.set_title("Happiness score boxplot by region", fontsize = 20)
ax.set_xlabel('Happiness Score', fontsize = 15)
ax.set_ylabel('Region', fontsize = 15)

# Order of regions
order = ["Australia and New Zealand", "Northern Europe", "Western Europe", 
         "Northern America", "Eastern Europe", "Southern Europe",
         "Latin America and the Caribbean",
         "Eastern Asia","Central Asia","South-eastern Asia","Western Asia",
         "Northern Africa","Southern Asia","Sub-Saharan Africa"]

# Create a boxplot with specified order of regions
sns.boxplot(data=full_data, y="sub_region", x="Ladder score", palette='Greens_r', order=order)

# Display the figure
plt.show()


# INTERPRETATION
# * Counties in northern europe are the happiest of all
# * Western Asia and Sub-Saharan Africa have wide range of happiness scores

# In[37]:


import plotly.express as px

# Define custom colors for the color scale
custom_color_scale = [
    (0.0, "red"),    # Low scores in red
    (0.5, "orange"), # Medium scores in orange
    (1.0, "green")   # High scores in green
]

fig = px.choropleth(full_data, locations="Country name", locationmode='country names',
                    color="Ladder score", hover_name="Country name",
                    title="World Happiness Report: Ladder score by country",
                    color_continuous_scale=custom_color_scale)

fig.show()


# In[38]:


import plotly.express as px

# Define custom colors for the color scale
custom_color_scale = [
    (0.0, "red"),    # Low scores in red
    (0.5, "orange"), # Medium scores in orange
    (1.0, "green")   # High scores in green
]

fig = px.choropleth(full_data, locations="Country name", locationmode='country names',
                    color="Ladder score", hover_name="Country name",projection = 'orthographic',
                    title="World Happiness Report: Ladder score by country",
                    color_continuous_scale='RdYlGn')

fig.show()


# In[39]:


import plotly.express as px

fig = px.scatter_geo(full_data, locations="Country name", locationmode='country names',
                    color="Ladder score", hover_name="Country name",
                    title="World Happiness Report: Ladder score by country",
                    color_continuous_scale=custom_color_scale, projection = "orthographic")
fig.show()


# 

# In[40]:


#Sunburst plot looking at the make up of the regions and sub_regions and how the happiness score of each country and sun_region compares with others in the same category

fig = px.sunburst(data_frame=full_data,
                  path=["region", "sub_region", "Country name"],
                  values="Ladder score",
                  color="Ladder score",
                  color_continuous_scale='RdYlGn',
                  width=800, 
                  height=800,
                  title = 'Happiness score sunburst - region / sub region / country')
fig.show()


# In[41]:


# Creating a correlation matrix
corrmat = full_data.corr()

# Plotting the heatmap
plt.figure(figsize=(10,8))
plt.title("Correlation heatmap", fontsize = 20)

sns.heatmap(corrmat ,annot=True, cmap="PiYG")

plt.show()


# In the correlation heatmap, positive correlations can be observed between:
# * Logged GDP per capita and the Ladder Score
# * Social support and the Ladder Score
# * Healthy life expectancy and the Ladder Score
# * Freedom to make life choices and the Ladder Score
# 
# Negative correlations can be observed between:
# * Generosity and the Ladder Score
# * Perceptions of corruption and the Ladder Score
# 
# This implies that as the Logged GDP per capita increases, as well as social support, healthy life expectancy, and freedom to make life choices, the Ladder Score also increases, increasing the happiness.
# 
# On the other hand, lower levels of generosity and higher levels of perceptions of corruption are associated with a lower Ladder Score, decreasing appiness.

# ---------------
# ### Takeaways:
# * The maximum Happiness Score is 7.8 of Finland.
# * Minimum Score is 1.8 which is of Afghanisthan
# * The Average Score for the countries is 5.5
# * GDP, Freedom, Life expectancy, Social Support, Freedom, region are directly related with happiness.
# * Generosity, perceptions of corruption do not have direct relation with Happiness.
# * Countries in the Europe are the happiest
# * Countries in African region are the least happiest
# * Countires in Asia have wide range of happiness score

# ---------------
# However, it is important to note that this comparison may not be entirely accurate, as factors such as economic conditions, access to education, healthcare, and political stability can also significantly influence happiness scores. Additionally, this data is based on an index and may not necessarily reflect absolute levels of happiness or well-being.

# ## STATISTICAL ANALYSIS 

# In[42]:


full_data.head()


# In[43]:


from scipy.stats import f_oneway

 
income = full_data['Logged GDP per capita']
family = full_data['Healthy life expectancy']
social = full_data['Social support']
health = full_data['Freedom to make life choices']
happiness = full_data['Ladder score']

# Perform the ANOVA test
#f_stat, p_value = foneway(income, family, social, health, happiness)
f_statistic, p_value = f_oneway(income, family, social, health, happiness)

print(f'F-Statistic: {f_statistic}')
print(f'P-Value: {p_value}')


# In[ ]:




