import streamlit as st 

def main():
	st.title("First app")
	st.write("Welcome to Streamlit")
	st.write("Here, you can display text, data, and visualization.")
	
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sns
	import plotly.express as px

	plt.style.use('ggplot')
	import plotly.io as pio
	pio.templates.default = "plotly_white"

	WHR = pd.read_csv('WHR2023.csv')
	country_mapping = pd.read_csv('continents2.csv')

	WHR.head()
	WHR.tail()
	WHR.describe()
	WHR.info()

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

	
	full_data.describe()

	top5 = WHR[['Country name', 'Ladder score']].head(5)
	print(top5)

	bottom5 = WHR[['Country name', 'Ladder score']].tail(5)
	print(bottom5)

	plt.subplot(2,1,1)
	sns.barplot(top5, x=top5['Ladder score'], y=top5['Country name'] ,color = 'green', edgecolor='white')

	plt.subplot(2,1, 2)
	sns.barplot(bottom5, x=bottom5['Ladder score'], y=bottom5['Country name'] ,color = 'red', edgecolor='white')

	plt.hist(WHR['Ladder score'], bins = 8 , edgecolor = 'white')

	plt.title('Happiness Score Distribution')
	plt.xlabel('Score')
	plt.ylabel('No of countries')



	plt.figure(figsize=[11, 7])

	plt.subplot(2,3,1)
	sns.scatterplot(data = full_data, x='Ladder score', y= 'Logged GDP per capita', hue = 'region')
	sns.regplot(data = full_data, x='Ladder score', y= 'Logged GDP per capita', scatter = False, ci=None)
	plt.xlabel('Happiness', fontsize=8)
	plt.ylabel('Logged GDP per capita', fontsize=8)
	plt.grid(True)

	plt.subplot(2,3,2)
	sns.scatterplot(data = full_data, x='Ladder score', y= 'Social support', hue = 'region')
	sns.regplot(data = full_data, x='Ladder score', y= 'Social support', scatter = False, ci=None)
	plt.xlabel('Happiness', fontsize=8)
	plt.ylabel('Social support', fontsize=8)

	plt.subplot(2,3,3)
	sns.scatterplot(data = full_data, x='Ladder score', y= 'Healthy life expectancy', hue = 'region')
	sns.regplot(data = full_data, x='Ladder score', y= 'Healthy life expectancy', scatter = False, ci=None)
	plt.xlabel('Happiness', fontsize=8)
	plt.ylabel('Healthy life expectancy', fontsize=8)

	plt.subplot(2,3,4)
	sns.scatterplot(data = full_data, x='Ladder score', y= 'Freedom to make life choices', hue = 'region')
	sns.regplot(data = full_data, x='Ladder score', y= 'Freedom to make life choices', scatter = False, ci=None)
	plt.xlabel('Happiness', fontsize=8)
	plt.ylabel('Freedom to make life choices', fontsize=8)

	plt.subplot(2,3,5)
	sns.scatterplot(data = full_data, x='Ladder score', y= 'Generosity', hue = 'region')
	sns.regplot(data = full_data, x='Ladder score', y= 'Generosity', scatter = False, ci=None)
	plt.xlabel('Happiness', fontsize=8)
	plt.ylabel('Generosity', fontsize=8)

	plt.subplot(2,3,6)
	sns.scatterplot(data = full_data, x='Ladder score', y= 'Perceptions of corruption', hue = 'region')
	sns.regplot(data = full_data, x='Ladder score', y= 'Perceptions of corruption', scatter = False, ci=None)
	plt.xlabel('Happiness', fontsize=8)
	plt.ylabel('Perceptions of corruption', fontsize=8)

	plt.tight_layout()
	plt.show()

	fig, ax = plt.subplots(figsize=(10,6))
	plt.title("Happiness score boxplot by region", fontsize = 20)
	sns.boxplot(data=full_data, y="region", x="Ladder score", palette='Greens_r', order=["Oceania", "Europe", "Americas", "Asia", "Africa"] ).set(
		xlabel='Happiness Score', 
		ylabel='Region'
	)
	plt.show()

	fig, ax = plt.subplots(figsize=(10,6))
	plt.title("Happiness score boxplot by region", fontsize = 20)
	ax.set_title("Happiness score boxplot by region", fontsize = 20)
	ax.set_xlabel('Happiness Score', fontsize = 15)
	ax.set_ylabel('Region', fontsize = 15)
	order = ["Australia and New Zealand", "Northern Europe", "Western Europe", 
			"Northern America", "Eastern Europe", "Southern Europe",
			"Latin America and the Caribbean",
			"Eastern Asia","Central Asia","South-eastern Asia","Western Asia",
			"Northern Africa","Southern Asia","Sub-Saharan Africa"]
	sns.boxplot(data=full_data, y="sub_region", x="Ladder score", palette='Greens_r', order=order)
	plt.show()

	import plotly.express as px
	custom_color_scale = [
		(0.0, "red"),
		(0.5, "orange"),
		(1.0, "green")
	]
	fig = px.choropleth(full_data, locations="Country name", locationmode='country names',
						color="Ladder score", hover_name="Country name",
						title="World Happiness Report: Ladder score by country",
						color_continuous_scale=custom_color_scale)
	fig.show()

	

if __name__ == "__main__":
	main()