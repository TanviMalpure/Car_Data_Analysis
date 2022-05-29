# Databricks notebook source
# DBTITLE 1,Technical Stack
# MAGIC %md
# MAGIC                                       
# MAGIC * Python - Coding Platform
# MAGIC * Pyspark - Distributed Framework
# MAGIC * SQL - Adhoc Data Analysis (Native Databricks database)

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Problem statement :
# MAGIC * Here I have used Data Analysis to identify which model can Tata launch and what could be the price of the car 

# COMMAND ----------

# DBTITLE 1,Factors considered for Analysis
# MAGIC %md
# MAGIC 
# MAGIC 1. Dimensions of the each body type based on Make of a vehicle
# MAGIC 2. Price - Make wise
# MAGIC 3. Correlation between Fuel vs Price
# MAGIC 4. Correlation between MPV Fuel_Type, Make & price 
# MAGIC 5. MPV Engine Displacement 
# MAGIC 6. MPV Engine Displacement VS Price
# MAGIC 7. Ertiga vs Eeco vs Xl6

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Assumptions: 
# MAGIC 
# MAGIC 1. I tried to find out the sales details of each Make & Variant over https://Kaggle.com & https://data.gov.in/ but could not find over the required granularity, so considering the sales details based on the current market growth of the segment & make in India.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Overview
# MAGIC 
# MAGIC This notebook will show how to create and query a table or DataFrame that is uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows to store data for querying inside of Databricks. This notebook assumes that there is already a file inside DBFS which we need to read.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, we can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/cars_engage_2022.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Data Cleansing Process

# COMMAND ----------

df.printSchema()

# COMMAND ----------

#Column heading Ex-Showroom_Price changed to Price
df = df.withColumnRenamed('Ex-Showroom_Price','Price') 

# COMMAND ----------

# Create a view or table

temp_table_name = "cars_engage_2022_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `cars_engage_2022_csv` 
# MAGIC where Price is Null

# COMMAND ----------

# DBTITLE 1,Casting datatype of Price column
from pyspark.sql.functions import *
from pyspark.sql.types import *
df = df.withColumn('Price',translate('Price','Rs.,','').cast(IntegerType()))

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,1. Dimensions of the each body type based on Make of a vehicle
dim_df = df.select('Make','Height','Length','Width','Body_Type','Price')

avg_dim_df = dim_df.groupBy('Make','Height','Length','Width','Body_Type').agg(avg('Price')) # Average out the Price irrespective of Variant

avg_dim_df = avg_dim_df.filter(col('Make').isNotNull()) # Removing Null values from Make column

# COMMAND ----------

display(avg_dim_df)

# COMMAND ----------

display(avg_dim_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### This analysis has been done to identify the vehicle segments where there is less competition. From this graph we can conclude Tata Motors do not have any vehicle in MPV segment. So, MPV could be a good choice to launch

# COMMAND ----------

display(avg_dim_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##### Most demanding cars in India

# COMMAND ----------

#I have filtered out most popular brands in India for further analysis
display(avg_dim_df.filter(col('Make').isin('Fiat','Hyundai','Nissan','Tata','Maruti Suzuki','Honda','Mahindra')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### From above analysis MPV and Coupe is a body type for which there is very less competition. Considering Indian scenario MPV is more popular than Coupe. So MPV has been shortlisted for further analysis.

# COMMAND ----------

# DBTITLE 1,MPV Fuel Type
mpv_df = df.filter(col('Body_Type')=='MPV')

mpv_clean = mpv_df.filter(col('Make').isNotNull()) # Filter null records 

# COMMAND ----------

# I have analyzed fuel type available in MPV segment. This analysis will help me in selecting appropriate Fuel Type for the vehicle to be launched
display(mpv_clean.select('Make','Model','Price', 'Fuel_Type').distinct().groupBy('Make','Model','Fuel_Type').agg(avg('Price').alias('Avg_Price')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### From this analysis it is concluded that Fuel type - Disesel will be appropriate for vehicle to be launched. Because there is less competition and there are only two vehicles(Ertiga and Marazo) available in this MPV segment

# COMMAND ----------

# DBTITLE 1,MPV Engine Displacement 
mpv_df_engine = df.filter(col('Body_Type')=='MPV')

mpv_clean_engine = mpv_df_engine.filter(col('Make').isNotNull()) # Filter null records 

mpv_clean_engine = mpv_clean_engine.withColumn('Displacement', translate('Displacement','c ,','').cast(IntegerType()))

# COMMAND ----------

#This analysis is to decide engine displacement.

# COMMAND ----------

display(mpv_clean_engine.select('Make','Model','Fuel_Type', 'Displacement').distinct().groupBy('Make','Model','Fuel_Type').agg(avg('Displacement').alias('Avg_Displacement')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### There are vehicle models available for the Engine Displacement of 1498. This analysis suggests that if we could decide to launch the vehicle with slighly less Engine Displacement e.g. 1250  then we will be able to launch the vehicle with slightly lower price than the existing fuel type - diesel models (Ertiga and Marazzo).

# COMMAND ----------

# DBTITLE 1,MPV Engine Displacement VS Price
# Below analysis is for the selecting price for the vehicle to be launched


def normalize_price(price):
  return price/1000

normalize_price_python = spark.udf.register('normalize_price_sql',normalize_price, DoubleType())

mpv_df_displacement_price = mpv_clean_engine.select('Make','Model','Fuel_Type', 'Displacement','Price').distinct().groupBy('Make','Model','Fuel_Type').agg(avg('Displacement').alias('Avg_Displacement'),avg('Price').alias('Avg_Price')).withColumn('Avg_Price',normalize_price_python('Avg_Price')).withColumn('Petrol_Price',when(col('Fuel_Type')=='Petrol',col('Avg_Price')).otherwise(lit(None))).withColumn('Diesel_Price',when(col('Fuel_Type')=='Diesel',col('Avg_Price')).otherwise(lit(None))).withColumn('CNG_Price',when(col('Fuel_Type')=='CNG',col('Avg_Price')).otherwise(lit(None))).withColumn('Petrol_Displacement',when(col('Fuel_Type')=='Petrol',col('Avg_Displacement')).otherwise(lit(None))).withColumn('Diesel_Displacement',when(col('Fuel_Type')=='Diesel',col('Avg_Displacement')).otherwise(lit(None))).withColumn('CNG_Displacement',when(col('Fuel_Type')=='CNG',col('Avg_Displacement')).otherwise(lit(None)))

# COMMAND ----------

display(mpv_df_displacement_price)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The price of the vehicle to be launched should be less than the price of Maruti Suzuki Ertiga Fuel type - Diesel, displacement - 1498
# MAGIC #### Price of top variant of Maruti Suzuki Ertiga Fuel type - Diesel and Engine Displacement of 1498 is INR 10.59 lakhs
# MAGIC #### So, Tata Motors should target forprice of INR 10.00 lakhs for the new vehicle to be launched

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Here I have merged two graphs line chart and bar graph together. Line chart shows relation between fuel type - displacement - vehicle model
# MAGIC #### Bar Graph shows relation between fuel type - price - vehicle model

# COMMAND ----------

import plotly.graph_objects as go

fig = go.Figure()

fig.update_layout(
    title="Displacement & Price Comparison for MPV",
    xaxis_title="MPV Vehicles",
    yaxis_title="CC & Price",
    )

fig.add_trace(
    go.Scatter(
        x=mpv_df_displacement_price.toPandas()['Model'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Model'].to_list()[0]],
        y=mpv_df_displacement_price.toPandas()['Petrol_Displacement'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Petrol_Displacement'].to_list()[0]],
      name="Petrol Displacement in CC" 
    ))
fig.add_trace(
    go.Scatter(
        x=mpv_df_displacement_price.toPandas()['Model'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Model'].to_list()[0]],
        y=mpv_df_displacement_price.toPandas()['Diesel_Displacement'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Diesel_Displacement'].to_list()[0]],
      name="Diesel Displacement in CC" 
    ))

fig.add_trace(
    go.Scatter(
        x=mpv_df_displacement_price.toPandas()['Model'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Model'].to_list()[0]],
        y=mpv_df_displacement_price.toPandas()['CNG_Displacement'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['CNG_Displacement'].to_list()[0]],
      name="CNG Displacement in CC" 
    ))

fig.add_trace(
    go.Bar(
        x=mpv_df_displacement_price.toPandas()['Model'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Model'].to_list()[0]],
        y=mpv_df_displacement_price.toPandas()['Petrol_Price'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Petrol_Price'].to_list()[0]],
      name="Petrol Price" 
    ))

fig.add_trace(
    go.Bar(
        x=mpv_df_displacement_price.toPandas()['Model'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Model'].to_list()[0]],
        y=mpv_df_displacement_price.toPandas()['Diesel_Price'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Diesel_Price'].to_list()[0]],
      name="Diesel Price" 
    ))

fig.add_trace(
    go.Bar(
        x=mpv_df_displacement_price.toPandas()['Model'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['Model'].to_list()[0]],
        y=mpv_df_displacement_price.toPandas()['CNG_Price'].to_list()[1:]+[mpv_df_displacement_price.toPandas()['CNG_Price'].to_list()[0]],
      name="CNG Price" 
    ))
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### I have performed this analysis to verify my previous analysis and shortlisting of – Engine displacement of 1250 CC and Price of INR 10 Lakhs for it. 

# COMMAND ----------

# DBTITLE 1,Ertiga vs Eeco vs Xl6
maruti_df = df.filter(col('Model').isin('Ertiga','Eeco','Xl6')).select('Make'
,'Model'
,'Price'
,'Fuel_Type'
,'Displacement'
,'Power_Steering'
,'Power_Windows'
,'Keyless_Entry'
,'Audiosystem'
,'Fasten_Seat_Belt_Warning'
,'Number_of_Airbags'
,'Turbocharger'
,'ISOFIX_(Child-Seat_Mount)'
,'Cruise_Control'
)

# COMMAND ----------

maruti_cleaned_df = maruti_df.withColumn('Power_Steering', when(col('Power_Steering').isNull(), 0).otherwise(1)).withColumn('Power_Windows', when(col('Power_Windows').isNull(), 0).otherwise(1)).withColumn('Keyless_Entry', when(col('Keyless_Entry').isNull(), 0).otherwise(1)).withColumn('Audiosystem', when(col('Audiosystem')=='Not on offer', 0).otherwise(1)).withColumn('Fasten_Seat_Belt_Warning', when(col('Fasten_Seat_Belt_Warning').isNull(), 0).otherwise(1)).withColumn('Number_of_Airbags', when(col('Number_of_Airbags').isNull(), 0).otherwise(col('Number_of_Airbags'))).withColumn('Turbocharger', when(col('Turbocharger').isNull(), 0).otherwise(1)).withColumn('ISOFIX_(Child-Seat_Mount)', when(col('ISOFIX_(Child-Seat_Mount)').isNull(), 0).otherwise(1)).withColumn('Cruise_Control', when(col('Cruise_Control').isNull(), 0).otherwise(1))

# COMMAND ----------

display(maruti_cleaned_df)

# COMMAND ----------

display(maruti_cleaned_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### I have categorized the features in 3 different categories - Comfort Feature, Regulatory_Essential Feature and Safety Feature

# COMMAND ----------

maruti_cleaned_agg_df = maruti_cleaned_df.groupBy('Make','Model','Fuel_Type').agg(max('Power_Steering').alias('Agg_Power_Steering'),max('Power_Windows').alias('Agg_Power_Windows'),max('Keyless_Entry').alias('Agg_Keyless_Entry'),max('Audiosystem').alias('Agg_Audiosystem'),max('Fasten_Seat_Belt_Warning').alias('Agg_Fasten_Seat_Belt_Warning'),max('Number_of_Airbags').alias('Agg_Number_of_Airbags'),max('Turbocharger').alias('Agg_Turbocharger'),max('ISOFIX_(Child-Seat_Mount)').alias('Agg_ISOFIX_Child-Seat_Mount'),max('Cruise_Control').alias('Agg_Cruise_Control'))

# COMMAND ----------

display(maruti_cleaned_agg_df)

# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC #### This analysis explains the vehicle model available in market and its feature content

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Following analysis is performed to show relation between Feature Type (Comfort, Safety, Regulatory_Essential) - Fuel Type - Vehicle Model

# COMMAND ----------

maruti_cleaned_agg_feature_df = maruti_cleaned_agg_df.withColumn('Regulatory_Essential_Feature',ceil(col('Agg_Power_Steering')+col('Agg_Power_Windows'))).withColumn('Safety_Feature',ceil(col('Agg_Fasten_Seat_Belt_Warning')+col('Agg_Number_of_Airbags')+col('Agg_ISOFIX_Child-Seat_Mount'))).withColumn('Comfort_Feature',ceil(col('Agg_Cruise_Control')+col('Agg_Turbocharger')+col('Agg_Keyless_Entry')+col('Agg_Audiosystem')))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### This graph shows relation between feature type (comfort feature, safety feature, Regulatory+Essential_Feature) - fule type - vehicle model

# COMMAND ----------

display(maruti_cleaned_agg_feature_df.select('Make','Model','Fuel_Type','Regulatory_Essential_Feature','Safety_Feature','Comfort_Feature'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Maruti Suzuki Xl6 - Petrol model highest number of Regulatory_Essential Features, highest number of Comfort Features and highest number of Regulatory Features.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Conclusion :
# MAGIC #### It was analysed and concluded in command 18 that Tata Motors should decide to launch MPV segment
# MAGIC #### It was analysed in command 26 that the Tata Motors should decide the Fuel Type Diesel
# MAGIC #### It was analysed in command 30 that the Tata Motors should decide the price of Engine Displacement of 1250 CC (less than the Ertiga and Marazzo)
# MAGIC #### It was analysed in command 33 that the Tata Motors should decide the price of INR 10 lakhs (less than the price of Maruti Suzuki Ertiga)
# MAGIC #### It is analysed in command 47 that Tata Motors should target all the features of Maruti Suzuki Xl6 to be considered for their new car to be launched
