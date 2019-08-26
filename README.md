# GroceryStore-Regression
Regression model to predict sales of grocery store

## Problem Statement

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.

Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

## Data
We have train (8523) and test (5681) data set, train data set has both input and output variable(s). You need to predict the sales for test data set.

<!DOCTYPE html>
<html>
<body>
<table style="width:100%">
  <tr>
    <td>Variable</td> <td>Description</td>
  </tr>
  <tr>
    <td>Item_Identifier</td><td>Unique product ID</td>
  </tr>
<tr>
  <td>Item_Weight</td><td>Weight of product</td>
  </tr>
<tr>
  <td>Item_Fat_Content</td><td>Whether the product is low fat or not</td>
  </tr>
  <tr>
    <td>Item_Visibility</td><td>The % of total display area of all products in a store allocated to the particular product</td>
  </tr>
<tr>
  <td>Item_Type</td><td>The category to which the product belongs</td>
  </tr>
<tr>
  <td>Item_MRP</td><td>Maximum Retail Price (list price) of the product</td>
  </tr>
<tr>
  <td>Outlet_Identifier</td><td>Unique store ID</td>
  </tr>
<tr>
  <td>Outlet_Establishment_Year</td><td>The year in which store was established</td>
  </tr>
<tr>
  <td>Outlet_Size</td><td>The size of the store in terms of ground area covered</td>
  </tr>
<tr>
  <td>Outlet_Location_Type</td><td>The type of city in which the store is located</td>
  </tr>
<tr>
  <td>Outlet_Type</td><td>Whether the outlet is just a grocery store or some sort of supermarket</td>
  </tr>
<tr>
  <td>Item_Outlet_Sales</td><td>Sales of the product in the particular store. This is the outcome variable to be predicted.</td>
  </tr>

</body>
</html>


