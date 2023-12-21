import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px


# Set the background color of Streamlit to white
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("url_goes_here")
    }
   .sidebar .sidebar-content {
        background: url("url_goes_here")
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("EDA on World Happiness Report")
body = '''
### INTRODUCTION
The annual World Happiness Report assesses global happiness by ranking countries based on factors like economic indicators, social support, life expectancy, freedom, generosity, and perceptions of corruption. The dataset for the 2023 report, available on Kaggle, provides valuable insights for governments and stakeholders to make informed decisions on improving the well-being of their citizens. Analyzing these factors helps identify areas for intervention and contributes to the ongoing dialogue on enhancing the quality of life worldwide.

### Objective:
The purpose of choosing this work aims to identify the key factors that contribute significantly to leading a more joyful life. Consequently, individuals and nations can prioritize these pivotal factors to attain elevated levels of happiness.
'''


citations = '''### Citation(Source):
* https://worldhappiness.report/ed/2023/
* https://www.kaggle.com/datasets/andradaolteanu/country-mapping-iso-continent-region
* https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2023tion = 


The following features were measured and examined:
1) Ladder score (Happiness score)	
2) Logged GDP per capita	
3) Social support	
4) Healthy life expectancy	
5) Freedom to make life choices
6) Generosity	
7) Perceptions of corruption	
8) Region
 '''

st.markdown(body, unsafe_allow_html=True)
st.markdown(citations , unsafe_allow_html=True)

st.markdown("### Snapshot of Datasets")
st.image(["imgs/s1.png","imgs/s2.png"], width = 400)


WHR = pd.read_csv('DataSets/WHR2023.csv')
country_mapping = pd.read_csv('DataSets/continents2.csv')

country_mapping.head()
country_mapping.info()

WHR.drop(WHR.loc[:, 'Standard error of ladder score':'lowerwhisker'].columns, inplace=True, axis=1)
WHR.drop(WHR.loc[:, 'Ladder score in Dystopia':'Dystopia + residual'].columns, inplace=True, axis=1)

country_mapping = country_mapping.drop(country_mapping.columns[[1,3,4,7,8,9,10]], axis = 1)

country_mapping = country_mapping.rename({'name':'Country name',
											'alpha-3':'iso_alpha',
											'sub-region':'sub_region'}, axis =1)

	
full_data = WHR.merge(country_mapping, on='Country name', how='left')
	

nan_region_rows = full_data[full_data['region'].isnull()]
	

full_data.dropna(subset=['region'], inplace=True)

# Sidebar title and text
#st.sidebar.title("First app")
#st.sidebar.write("Welcome to Streamlit")
#st.sidebar.write("Here, you can display text, data, and visualization.")

# Display top and bottom 5 countries
st.subheader("Top 5 Happiest Countries")
st.write(WHR[['Country name', 'Ladder score']].head(5))

st.subheader("Bottom 5 Happiest Countries")
st.write(WHR[['Country name', 'Ladder score']].tail(5))

# Display bar plots
st.subheader("Top 5 and Bottom 5 Happiness Scores")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
sns.barplot(data=WHR[['Country name', 'Ladder score']].head(5), x='Ladder score', y='Country name', color='green', ax=ax1)
sns.barplot(data=WHR[['Country name', 'Ladder score']].tail(5), x='Ladder score', y='Country name', color='red', ax=ax2)
st.pyplot(fig)

# Display histogram
st.subheader("Happiness Score Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(WHR['Ladder score'], bins=8, edgecolor='white')
plt.title('Happiness Score Distribution')
plt.xlabel('Score')
plt.ylabel('No of countries')
st.pyplot(fig)

interpret1 = '''
INTERPRETATION:
* We can see that most of the countries are concenrated near to score 6
* There are least amount of countries with ladder score between 1 and 3
'''
st.write(interpret1)

# Display scatter plots
st.subheader("Scatter Plots")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.scatterplot(data=full_data, x='Ladder score', y='Logged GDP per capita', hue='region', ax=axes[0, 0])
sns.scatterplot(data=full_data, x='Ladder score', y='Social support', hue='region', ax=axes[0, 1])
sns.scatterplot(data=full_data, x='Ladder score', y='Healthy life expectancy', hue='region', ax=axes[0, 2])
sns.scatterplot(data=full_data, x='Ladder score', y='Freedom to make life choices', hue='region', ax=axes[1, 0])
sns.scatterplot(data=full_data, x='Ladder score', y='Generosity', hue='region', ax=axes[1, 1])
sns.scatterplot(data=full_data, x='Ladder score', y='Perceptions of corruption', hue='region', ax=axes[1, 2])
fig.tight_layout()
st.pyplot(fig)

interpret2 = '''
INTERPRETATION:
* GDP, Freedom, Life expectancy, Social Support, Freedom are directly related with happiness.
* Generosity, perceptions of corruption do not have direct relation with Happiness.
'''
st.write(interpret2)


# Plotting each happiness factor against Ladder score with subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))

# Plot 1
sns.lineplot(data=full_data, x='Ladder score', y='Logged GDP per capita', hue='region', ax=axes[0, 0])
axes[0, 0].set(xlabel='Happiness', ylabel='Logged GDP per capita')
axes[0, 0].grid(True)

# Plot 2
sns.lineplot(data=full_data, x='Ladder score', y='Social support', hue='region', ax=axes[0, 1])
axes[0, 1].set(xlabel='Happiness', ylabel='Social support')

# Plot 3
sns.lineplot(data=full_data, x='Ladder score', y='Healthy life expectancy', hue='region', ax=axes[1, 0])
axes[1, 0].set(xlabel='Happiness', ylabel='Healthy life expectancy')

# Plot 4
sns.lineplot(data=full_data, x='Ladder score', y='Freedom to make life choices', hue='region', ax=axes[1, 1])
axes[1, 1].set(xlabel='Happiness', ylabel='Freedom to make life choices')

# Plot 5
sns.lineplot(data=full_data, x='Ladder score', y='Generosity', hue='region', ax=axes[2, 0])
axes[2, 0].set(xlabel='Happiness', ylabel='Generosity')

# Plot 6
sns.lineplot(data=full_data, x='Ladder score', y='Perceptions of corruption', hue='region', ax=axes[2, 1])
axes[2, 1].set(xlabel='Happiness', ylabel='Perceptions of corruption')

# Adjust layout
plt.tight_layout()

# Display the plots using Streamlit
st.pyplot(fig)





# Display box plots
st.subheader("Box Plots by Region")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=full_data, y="region", x="Ladder score", palette='Greens_r', order=["Oceania", "Europe", "Americas", "Asia", "Africa"], ax=ax)
ax.set(xlabel='Happiness Score', ylabel='Region')
st.pyplot(fig)

interpret3 = '''
INTERPRETATION

* The plot suggests that there is not a significant gap between the most and least happy nations.

* Within each region, there is a mix of extremely happy and extremely unhappy individuals.

* Asia and Africa, have a more diverse range of happiness scores compared to others like Oceania, Europe, and Americas.

* The boxplot for the region "Oceania" is shorter in length compared to other regions, indicating that happiness scores in Oceania are more concentrated and less spread out.

* The boxplot for "Asia" has the largest length, indicating that happiness scores in Asia are the most spread out, with a larger range of values.

* The boxplot for "Africa" is relatively short in length, suggesting that happiness scores in Africa are also more concentrated and less spread out compared to other regions.
'''
st.write(interpret3)




# Streamlit App
st.title("Happiness Score Boxplot by Region")

# Create a figure with specified size
fig, ax = plt.subplots(figsize=(10, 6))

# Set title and labels for x and y axes
ax.set_title("Happiness score boxplot by region", fontsize=20)
ax.set_xlabel('Happiness Score', fontsize=15)
ax.set_ylabel('Region', fontsize=15)

# Order of regions
order = ["Australia and New Zealand", "Northern Europe", "Western Europe",
         "Northern America", "Eastern Europe", "Southern Europe",
         "Latin America and the Caribbean",
         "Eastern Asia", "Central Asia", "South-eastern Asia", "Western Asia",
         "Northern Africa", "Southern Asia", "Sub-Saharan Africa"]

# Create a boxplot with specified order of regions
sns.boxplot(data=full_data, y="sub_region", x="Ladder score", palette='Greens_r', order=order)

# Display the figure in Streamlit
st.pyplot()

interpret4 = '''INTERPRETATION
* Counties in northern europe are the happiest of all
* Western Asia and Sub-Saharan Africa have wide range of happiness scores
'''
st.write(interpret4)



# Display choropleth map
st.subheader("Choropleth Map of Happiness Scores by Country")
custom_color_scale = [(0.0, "red"), (0.5, "orange"), (1.0, "green")]
fig = px.choropleth(full_data, locations="Country name", locationmode='country names',
                    color="Ladder score", hover_name="Country name",
                    title="World Happiness Report: Ladder score by country",
                    color_continuous_scale=custom_color_scale)
st.plotly_chart(fig)

# Choropleth Map of Happiness Scores by Country
st.subheader("Choropleth Map of Happiness Scores by Country")
fig = px.choropleth(full_data, 
                    locations="Country name", 
                    locationmode='country names',
                    color="Ladder score", 
                    hover_name="Country name",
                    projection='orthographic',  # Use an orthographic projection
                    title="World Happiness Report: Ladder score by country",
                    color_continuous_scale=custom_color_scale)
st.plotly_chart(fig)


# Sunburst Chart of Happiness Scores by Region, Sub-Region, and Country
st.subheader("Happiness Score Sunburst - Region / Sub-Region / Country")

fig = px.sunburst(data_frame=full_data,
                  path=["region", "sub_region", "Country name"],
                  values="Ladder score",
                  color="Ladder score",
                  color_continuous_scale='RdYlGn',
                  width=800, 
                  height=800,
                  title='Happiness Score Sunburst - Region / Sub-Region / Country')

st.plotly_chart(fig)



# Creating a correlation matrix
corrmat = full_data.corr()

# Streamlit App
st.title("Correlation Heatmap")

# Plotting the heatmap using Seaborn
plt.figure(figsize=(10, 8))
plt.title("Correlation heatmap", fontsize=20)

sns.heatmap(corrmat, annot=True, cmap="PiYG", fmt=".2f", linewidths=0.5)

# Display the heatmap in Streamlit
st.pyplot()

interpret4 = '''In the correlation heatmap, positive correlations can be observed between:
* Logged GDP per capita and the Ladder Score
* Social support and the Ladder Score
* Healthy life expectancy and the Ladder Score
* Freedom to make life choices and the Ladder Score

Negative correlations can be observed between:
* Generosity and the Ladder Score
* Perceptions of corruption and the Ladder Score

This implies that as the Logged GDP per capita increases, as well as social support, healthy life expectancy, and freedom to make life choices, the Ladder Score also increases, increasing the happiness.

On the other hand, lower levels of generosity and higher levels of perceptions of corruption are associated with a lower Ladder Score, decreasing appiness.
'''
st.write(interpret4)

conclusion = '''
---------------
### Takeaways:
* The maximum Happiness Score is 7.8 of Finland.
* Minimum Score is 1.8 which is of Afghanisthan
* The Average Score for the countries is 5.5
* GDP, Freedom, Life expectancy, Social Support, Freedom, region are directly related with happiness.
* Generosity, perceptions of corruption do not have direct relation with Happiness.
* Countries in the Europe are the happiest
* Countries in African region are the least happiest
* Countires in Asia have wide range of happiness score

---------------
However, it is important to note that this comparison may not be entirely accurate, as factors such as economic conditions, access to education, healthcare, and political stability can also significantly influence happiness scores. Additionally, this data is based on an index and may not necessarily reflect absolute levels of happiness or well-being.
'''
st.markdown(conclusion)
