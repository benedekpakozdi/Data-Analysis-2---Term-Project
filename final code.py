import pandas as pn 
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer



df = pn.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

#lets check the variables dependent: Price

plt.hist(df['selling_price'], bins=range(0, 900000, 20000), edgecolor='k')
plt.xlabel('Prices')
plt.ylabel('Number of cars')
plt.show()
#funny to see that people like to state their prices so it is close to some 'full' number (100 or 50)

################ Explanatory variables ###########################
#'km_driven', 'fuel', 'seller_type','transmission', 'owner'#

#1.: Owner

df['owner'].value_counts()

#will need to transform it into amount now its First, Second etc


#2.: year

df['year'].value_counts()

        # Oldest 1992, newest 2017 

plt.hist(df['year'], bins=range(1992, 2017, 3), edgecolor='k')
plt.xlabel('Year of production')
plt.ylabel('Number of cars')
plt.show()

# nice logaritmic shape

#3.: KM driven

df['km_driven'].describe()

plt.hist(df['km_driven'], bins=range(1, 300000, 10000), edgecolor='k')
plt.xlabel('mileage')
plt.ylabel('Number of cars')
plt.show()

plt.hist(df['km_driven'], bins=range(300000, 806599, 100000), edgecolor='k')
plt.xlabel('mileage')
plt.ylabel('Number of cars (over 300,000KM)')
plt.show()

# migth worth to drop cars over 300,000 --> X value so worth it

#4.: fuel

df['fuel'].value_counts()

# there are a few cars with gas (CNG, LPG, one electric)

#5.: transmission 

df['transmission'].value_counts()
#mostly manuals only a few automaic --> however automatics are genually more expensive

df['seller_type'].value_counts()
    #Individual          3244
    #Dealer               994
    #Trustmark Dealer     102
        #trusted dealer might have an advantage to price higher 

########relation between the explanatory variables

x = df['year']
y = df['km_driven']

# Fit linear regression (degree=1)
m, b = np.polyfit(x, y, 1)

plt.figure(figsize=(10,5))
plt.scatter(x, y, alpha=0.2)

# Regression line
plt.plot(x, m*x + b, linewidth=2)

plt.xlabel("Year")
plt.ylabel("KM Driven")
plt.title("Year vs KM Driven with Regression Line")
plt.show()

##should check###

year_mileage = smf.ols(formula = 'km_driven ~ year', data=df).fit(cov_type='HC3')
print(year_mileage.summary())

#quite low r square  0.176 --> it will not neccesarily cause mulicolinerity --> lot of observations and robust standard errors it will be fine

df['log_mileage'] = np.log(df['km_driven'])

# --> checked for possible research questions that meets the requirement and not has been analysed yet

############################################################################
#################### DATA CLEANING #########################################
############################################################################

df.dropna()

# drop unneccesary variables (year,seller_type,owner)

keep_cols = [
    'selling_price', 
    'year',
    'name',   
    'km_driven',  
    'fuel', 
    'transmission',
    'log_mileage',
]

df_clean = df[keep_cols].copy()

# as we are interested in Petrol and Disel cars we can drop other fuel types and create a dummy for disel-s

df_clean = df_clean[df_clean['fuel'].isin(['Diesel', 'Petrol'])]

df_clean['diesel'] = (df_clean['fuel'] == 'Diesel').astype(int)

# create a dummy for automatic transmissions 

df_clean['automatic'] = (df_clean['transmission'] == 'Automatic').astype(int)

df_clean['log_price'] = np.log(df_clean['selling_price'])

plt.hist(df_clean['log_price'], bins=range(10, 20, 1), edgecolor='k')
plt.xlabel('log-Prices')
plt.ylabel('Number of cars')
plt.show()

### better to use log prices --> normaly distributied not right screwed 

# last but not least create age variable --> higher is worst --more easy to interpret (2022 is the base year as the dataset was uploaded 3 years ago)

df_clean['age'] = 2022 - df_clean['year'] 

########################### df_clean2 #################
########## issues with km_driven - many exterme values, lets make a new dataset where these are droped: 

#as i showed there are exterme x values that can distort the results:
plt.hist(df_clean['log_mileage'], bins=range(10, 20, 1), edgecolor='k')
plt.xlabel('log-mileage')
plt.ylabel('Number of cars')
plt.show()

# try to drop values above 150,000 km as higher maintanence cost are must occure above this distance --> around 5th oil change period (you should change oil in each 30k km)

df_clean2 = df_clean.copy()

df_clean2 = df_clean2[df_clean2['km_driven'] <= 150000]

df_clean2['log_mileage'] = np.log(df_clean2['km_driven'])


###################################################################
######################### Analysis ################################
###################################################################

# note on notation: m ~ model
#             first number: 1 ~ base model, 2 ~ full model, 3 ~ robustness
#             second number ~ variant of the model
#             third number ~ 1 if unfiltered/complete dataset used
#       e.g: m231 ~ full model, 3rd variant, unfiltered dataset used (df_clean)  



m111 = smf.ols(formula = 'selling_price ~ km_driven', data=df_clean).fit(cov_type='HC3')
print(m111.summary())

#low r square 0.036

m121 = smf.ols(formula = 'log_price ~ km_driven', data=df_clean).fit(cov_type='HC3')
print(m121.summary())

#still low r square 0.054

m131 = smf.ols(formula = 'log_price ~ log_mileage', data=df_clean).fit(cov_type='HC3')
print(m131.summary())

#slightly higher 0.84

############# SECOND, fliltered dataframe###########
#lets do the same but with the second dataframe (mileage is maximized in 150k km)


m11 = smf.ols(formula = 'selling_price ~ km_driven', data=df_clean2).fit(cov_type='HC3')
print(m11.summary())


m12 = smf.ols(formula = 'log_price ~ km_driven', data=df_clean2).fit(cov_type='HC3')
print(m12.summary())


m13 = smf.ols(formula = 'log_price ~ log_mileage', data=df_clean2).fit(cov_type='HC3')
print(m13.summary())

#slightly better results adjusted rsquare (0.089)

############# now lets see the full model ###########

# with log - log 

m21 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_clean2).fit(cov_type='HC3')
print(m21.summary())

m211 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_clean).fit(cov_type='HC3')
print(m211.summary())

# with log price - level distance 

m22 = smf.ols(formula = 'log_price ~ km_driven + diesel '
'+ km_driven:diesel + automatic + km_driven:automatic + age', data=df_clean2).fit(cov_type='HC3')
print(m22.summary())

m221 = smf.ols(formula = 'log_price ~ km_driven + diesel '
'+ km_driven:diesel + automatic + km_driven:automatic + age', data=df_clean).fit(cov_type='HC3')
print(m221.summary())

#with level price - level distance

m23 = smf.ols(formula = 'selling_price ~ km_driven + diesel '
'+ km_driven:diesel + automatic + km_driven:automatic + age', data=df_clean2).fit(cov_type='HC3')
print(m23.summary())

m231 = smf.ols(formula = 'selling_price ~ km_driven + diesel '
'+ km_driven:diesel + automatic + km_driven:automatic + age', data=df_clean).fit(cov_type='HC3')
print(m231.summary())

#with level price - log distance

m24 = smf.ols(formula = 'selling_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_clean2).fit(cov_type='HC3')
print(m24.summary())

m241 = smf.ols(formula = 'selling_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_clean).fit(cov_type='HC3')
print(m241.summary())


###############################################################################################
############################## Robustness check ###############################################
###############################################################################################

#######IDEA 1:
###

# omited variables bias --> we don't know the original price of the cars

df_robustness = df_clean.copy()

df_robustness['brand'] = df_robustness['name'].str.split().str[0]

df_robustness['brand'].unique()

# differnt cars: 'Maruti', 'Hyundai', 'Datsun', 'Honda', 'Tata', 'Chevrolet',
#       'Toyota', 'Jaguar', 'Mercedes-Benz', 'Audi', 'Skoda', 'Jeep',
#       'BMW', 'Mahindra', 'Ford', 'Nissan', 'Renault', 'Fiat',
#       'Volkswagen', 'Volvo', 'Mitsubishi', 'Land', 'Daewoo', 'MG',
#       'Force', 'Isuzu', 'OpelCorsa', 'Ambassador', 'Kia'

# there are cars that are more expensive as they position themselves as a 'premium' brands. e.g.: BMW, Mercedes etc

#there are: Jaguar, Mercedes-Benz, Audi, Jeep, BMW, Volvo, Land (land rover) <-- these must be more expensive when new


premium_brands = ['Jaguar', 'Mercedes-Benz', 'Audi', 'Jeep', 'BMW', 'Volvo', 'Land']

df_robustness['premium'] = df_robustness['brand'].apply(lambda x: 1 if x in premium_brands else 0)
df_robustness['non_premium'] = df_robustness['brand'].apply(lambda x: 0 if x in premium_brands else 1)

# issue --> only a few premium cars are run in the indian market (we can try to make the estimates)

df_robustness_premium = df_robustness[df_robustness['premium'] == 1].copy()

m31 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_robustness_premium).fit(cov_type='HC3')
print(m31.summary())

df_robustness_nonpremium = df_robustness[df_robustness['premium'] == 0].copy()

m32 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_robustness_nonpremium).fit(cov_type='HC3')
print(m32.summary())

# lets create more categories:

utility = ['Toyota', 'Mahindra', 'Isuzu', 'Force', 'Mitsubishi']
mid_range = ['Honda', 'Volkswagen', 'Ford', 'Skoda', 'Nissan', 'Renault', 'Kia', 'MG', 'Kia']
mass_market = ['Maruti', 'Hyundai', 'Tata', 'Datsun', 'Chevrolet', 'Fiat', 'Daewoo', 'OpelCorsa', 'Ambassador']

df_robustness['unility'] = df_robustness['brand'].apply(lambda x: 1 if x in utility else 0)
df_robustness['mid_range'] = df_robustness['brand'].apply(lambda x: 1 if x in mid_range else 0)
df_robustness['mass_market'] = df_robustness['brand'].apply(lambda x: 1 if x in mass_market else 0)

df_robustness_utility = df_robustness[df_robustness['unility'] == 1].copy()
df_robustness_mid_range = df_robustness[df_robustness['mid_range'] == 1].copy()
df_robustness_mass_market = df_robustness[df_robustness['mass_market'] == 1].copy()

m33 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_robustness_utility).fit(cov_type='HC3')
print(m33.summary())

m34 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_robustness_mid_range).fit(cov_type='HC3')
print(m34.summary())

m35 = smf.ols(formula = 'log_price ~ log_mileage + diesel '
'+ log_mileage:diesel + automatic + log_mileage:automatic + age', data=df_robustness_mass_market).fit(cov_type='HC3')
print(m35.summary())


###############################################################
################# now nice plots ##############################
###############################################################

### filtered data: 

stargazer1 = Stargazer([m21, m22, m23, m24])

stargazer1.title("Regression Results – Models using dataset with mileage below 300,000km")

stargazer1.custom_columns(
    ["log(price)", "log(price)", "price", "price"],
    [1, 1, 1, 1]
)

# Remove the single-DV label (workaround for older Stargazer)
stargazer1.dependent_variable_name("")

html1 = stargazer1.render_html()

with open("regression_table_clean.html", "w", encoding="utf-8") as f:
    f.write(html1)

### complete data:

stargazer2 = Stargazer([m211, m221, m231, m241])

stargazer2.title("Regression Results – Models using complete dataset")

stargazer2.custom_columns(
    ["log(price)", "log(price)", "price", "price"],
    [1, 1, 1, 1]
)

stargazer2.dependent_variable_name("")

html2 = stargazer2.render_html()

with open("regression_table_clean2.html", "w", encoding="utf-8") as f:
    f.write(html2)

### main data


stargazer3 = Stargazer([m21, m211])

stargazer3.title("Regression Results – Log - Log model")

stargazer3.custom_columns(
    ["log(price) Cleaned data", "log(price) Full data"],
    [1, 1]
)

stargazer3.dependent_variable_name("")

html3 = stargazer3.render_html()

with open("regression_table_main.html", "w", encoding="utf-8") as f:
    f.write(html3)


##### robostness

stargazer4 = Stargazer([m31, m32])

stargazer4.title("Robostness based on new price")

stargazer4.custom_columns(
    ["log(price) Premium", "log(price) Non-premium"],
    [1, 1]
)

stargazer4.dependent_variable_name("")

html4 = stargazer4.render_html()

with open("regression_table_robostness1.html", "w", encoding="utf-8") as f:
    f.write(html4)

##### robustness 2

stargazer5 = Stargazer([m31, m33, m34, m35])

stargazer5.title("Robostness based categories")

stargazer5.custom_columns(
    ["log(price) Premium", "log(price) Utility", "log(price) Mid-Range", "log(price) Mass-Market"],
    [1, 1, 1, 1]
)

stargazer5.dependent_variable_name("")

html5 = stargazer5.render_html()

with open("regression_table_robostness2.html", "w", encoding="utf-8") as f:
    f.write(html5)
