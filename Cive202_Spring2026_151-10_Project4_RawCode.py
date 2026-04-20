#!/usr/bin/env python
# coding: utf-8

# PROJECT 4
# Oklahoma and Washington Risk Comparison

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd


# In[3]:


## Load files from the folder
tracts = gpd.read_file("NRI_Shapefile_CensusTracts.shp")

ok_tract_data = pd.read_csv("NRI_Table_CensusTracts_Oklahoma.csv", low_memory=False)
wa_tract_data = pd.read_csv("NRI_Table_CensusTracts_Washington.csv", low_memory=False)

ok_county_data = pd.read_csv("NRI_Table_Counties_Oklahoma.csv", low_memory=False)
wa_county_data = pd.read_csv("NRI_Table_Counties_Washington.csv", low_memory=False)

nri_dictionary = pd.read_csv("NRIDataDictionary.csv", low_memory=False)

print("Shapefile loaded.")
print("Oklahoma tract file shape:", ok_tract_data.shape)
print("Washington tract file shape:", wa_tract_data.shape)


# In[5]:


## Clean Shapefile
tracts = tracts.replace(-9999, np.nan)

print("Original CRS:", tracts.crs)

if tracts.crs is None:
    tracts = tracts.set_crs(epsg=3857)

tracts = tracts.to_crs(epsg=4326)

print("New CRS:", tracts.crs)
tracts.head()


# In[7]:


## Celan seperate tracts head
ok_tract_data = ok_tract_data.replace(-9999, np.nan)
wa_tract_data = wa_tract_data.replace(-9999, np.nan)

ok_tract_data["STATE_SOURCE"] = "Oklahoma"
wa_tract_data["STATE_SOURCE"] = "Washington"

state_tract_data = pd.concat([ok_tract_data, wa_tract_data], ignore_index=True)

print(state_tract_data.shape)
state_tract_data.head()


# In[8]:


## Filter shapefile to Oklahoma and Washington 
tracts_ok = tracts[tracts["STATE"] == "Oklahoma"].copy()
tracts_wa = tracts[tracts["STATE"] == "Washington"].copy()

tracts_two_states = pd.concat([tracts_ok, tracts_wa], ignore_index=True)

print(tracts_two_states["STATE"].value_counts())
print(tracts_two_states.shape)


# In[11]:


## Clean join columns
tracts_two_states["TRACTFIPS"] = (
    tracts_two_states["TRACTFIPS"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(11)
)

state_tract_data["FIPS"] = (
    state_tract_data["FIPS"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(11)
)

print(tracts_two_states["TRACTFIPS"].head())
print(state_tract_data["FIPS"].head())


# In[12]:


## Merge NRI shapefile with SVI tract data
risk_data = tracts_two_states.merge(
    state_tract_data,
    left_on="TRACTFIPS",
    right_on="FIPS",
    how="left",
    suffixes=("_nri", "_svi")
)

print(risk_data.shape)
risk_data.head()


# In[25]:


## Make one clean state column
if "STATE_nri" in risk_data.columns:
    risk_data["STATE"] = risk_data["STATE_nri"]
elif "STATE_x" in risk_data.columns:
    risk_data["STATE"] = risk_data["STATE_x"]
elif "STATE_shape" in risk_data.columns:
    risk_data["STATE"] = risk_data["STATE_shape"]

print([col for col in risk_data.columns if "STATE" in col])


# In[22]:


## Main NRI columns from the shapefile
risk_score_col = "RISK_SCORE"
risk_spctl_col = "RISK_SPCTL"
eal_col = "EAL_VALT"
population_col = "POPULATION"

print("NRI score column:", risk_score_col)
print("NRI percentile column:", risk_spctl_col)
print("Impact column:", eal_col)
print("Population column:", population_col)


# In[14]:


## Check Missing Values
missing_counts = risk_data.isna().sum().sort_values(ascending=False)
missing_counts = missing_counts[missing_counts > 0]

missing_counts.head(25)


# In[39]:


## Helper functions
def min_max_scale(series):
    series = pd.to_numeric(series, errors="coerce")
    min_val = series.min()
    max_val = series.max()

    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(np.nan, index=series.index)

    return (series - min_val) / (max_val - min_val)


def risk_rating_from_percentile(series):
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=[-0.01, 20, 40, 60, 80, 100],
        labels=["Very Low", "Low", "Moderate", "High", "Very High"]
    )


def first_column_that_exists(df, column_list):
    for col in column_list:
        if col in df.columns:
            return col
    return None


# In[20]:


## Find the main comparison columns
risk_score_col = first_column_that_exists(risk_data, ["RISK_SCORE_shape", "RISK_SCORE_tract", "RISK_SCORE"])
risk_spctl_col = first_column_that_exists(risk_data, ["RISK_SPCTL_shape", "RISK_SPCTL_tract", "RISK_SPCTL"])
eal_col = first_column_that_exists(risk_data, ["EAL_VALT_shape", "EAL_VALT_tract", "EAL_VALT"])
population_col = first_column_that_exists(risk_data, ["POPULATION_shape", "POPULATION_tract", "POPULATION"])

print("NRI score column:", risk_score_col)
print("NRI percentile column:", risk_spctl_col)
print("Impact column:", eal_col)
print("Population column:", population_col)


# In[40]:


## Composite alternative risk
afreq_cols = [col for col in risk_data.columns if col.endswith("_AFREQ")]

risk_data["total_frequency"] = risk_data[afreq_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
risk_data["scaled_total_frequency"] = min_max_scale(risk_data["total_frequency"])
risk_data["scaled_eal_total"] = min_max_scale(risk_data[eal_col])

risk_data["alt_risk_composite"] = risk_data["scaled_total_frequency"] * risk_data["scaled_eal_total"]
risk_data["alt_risk_composite_pct"] = risk_data["alt_risk_composite"].rank(pct=True) * 100
risk_data["alt_risk_composite_rating"] = risk_rating_from_percentile(risk_data["alt_risk_composite_pct"])

risk_data[[
    "STATE",
    "TRACTFIPS",
    eal_col,
    "total_frequency",
    "alt_risk_composite",
    "alt_risk_composite_pct",
    "alt_risk_composite_rating"
]].head()


# In[28]:


## Per-capita alternative risk
risk_data["population_clean"] = pd.to_numeric(risk_data[population_col], errors="coerce")
risk_data.loc[risk_data["population_clean"] == 0, "population_clean"] = np.nan

risk_data["alt_risk_per_capita"] = risk_data["alt_risk_composite"] / risk_data["population_clean"]
risk_data["alt_risk_per_capita_pct"] = risk_data["alt_risk_per_capita"].rank(pct=True) * 100
risk_data["alt_risk_per_capita_rating"] = risk_rating_from_percentile(risk_data["alt_risk_per_capita_pct"])

risk_data[[
    "STATE",
    "TRACTFIPS",
    "population_clean",
    "alt_risk_per_capita",
    "alt_risk_per_capita_pct",
    "alt_risk_per_capita_rating"
]].head()


# In[29]:


## Find hazard-specific columns
ok_tornado_freq = first_column_that_exists(risk_data, ["TRND_AFREQ", "TORN_AFREQ"])
ok_tornado_impact = first_column_that_exists(risk_data, ["TRND_EALT", "TORN_EALT"])

wa_eq_freq = first_column_that_exists(risk_data, ["ERQK_AFREQ"])
wa_eq_impact = first_column_that_exists(risk_data, ["ERQK_EALT"])

wa_fire_freq = first_column_that_exists(risk_data, ["WFIR_AFREQ", "WILD_AFREQ"])
wa_fire_impact = first_column_that_exists(risk_data, ["WFIR_EALT", "WILD_EALT"])

print("Oklahoma tornado columns:", ok_tornado_freq, ok_tornado_impact)
print("Washington earthquake columns:", wa_eq_freq, wa_eq_impact)
print("Washington wildfire columns:", wa_fire_freq, wa_fire_impact)


# In[45]:


## Oklahoma tornado alternative risk
risk_data["ok_tornado_freq_scaled"] = min_max_scale(risk_data["TRND_AFREQ"])
risk_data["ok_tornado_impact_scaled"] = min_max_scale(risk_data["TRND_EALT"])
risk_data["ok_tornado_alt_risk"] = (
    risk_data["ok_tornado_freq_scaled"] * risk_data["ok_tornado_impact_scaled"]
)
risk_data["ok_tornado_alt_risk_pct"] = risk_data["ok_tornado_alt_risk"].rank(pct=True) * 100


# In[46]:


## Washington earthquake alternative risk
risk_data["wa_earthquake_freq_scaled"] = min_max_scale(risk_data["ERQK_AFREQ"])
risk_data["wa_earthquake_impact_scaled"] = min_max_scale(risk_data["ERQK_EALT"])
risk_data["wa_earthquake_alt_risk"] = (
    risk_data["wa_earthquake_freq_scaled"] * risk_data["wa_earthquake_impact_scaled"]
)
risk_data["wa_earthquake_alt_risk_pct"] = risk_data["wa_earthquake_alt_risk"].rank(pct=True) * 100


# In[47]:


## Washington wildfire alternative risk
risk_data["wa_wildfire_freq_scaled"] = min_max_scale(risk_data["WFIR_AFREQ"])
risk_data["wa_wildfire_impact_scaled"] = min_max_scale(risk_data["WFIR_EALT"])
risk_data["wa_wildfire_alt_risk"] = (
    risk_data["wa_wildfire_freq_scaled"] * risk_data["wa_wildfire_impact_scaled"]
)
risk_data["wa_wildfire_alt_risk_pct"] = risk_data["wa_wildfire_alt_risk"].rank(pct=True) * 100


# In[48]:


## Summary Table 1: State-level comparison of the NRI and proposed risk

summary_table_1 = risk_data.groupby("STATE").agg(
    mean_nri_risk_score=(risk_score_col, "mean"),
    mean_nri_percentile=(risk_spctl_col, "mean"),
    mean_alt_composite=("alt_risk_composite", "mean"),
    mean_alt_composite_pct=("alt_risk_composite_pct", "mean"),
    median_alt_composite=("alt_risk_composite", "median"),
    tract_count=("TRACTFIPS", "count")
).round(4)

summary_table_1


# In[49]:


## Summary Table 2: Category comparison between NRI and proposed risk
risk_data["nri_rating_from_percentile"] = risk_rating_from_percentile(risk_data[risk_spctl_col])

summary_table_2 = pd.crosstab(
    risk_data["nri_rating_from_percentile"],
    risk_data["alt_risk_composite_rating"],
    dropna=False
)

summary_table_2


# In[51]:


## Summary Table 3: Hazard-specific alternative risk comparison

hazard_rows = []

ok_only = risk_data[risk_data["STATE"] == "Oklahoma"]
wa_only = risk_data[risk_data["STATE"] == "Washington"]

hazard_rows.append({
    "State": "Oklahoma",
    "Hazard": "Tornado",
    "Mean_Alternative_Risk": ok_only["ok_tornado_alt_risk"].mean(),
    "Median_Alternative_Risk": ok_only["ok_tornado_alt_risk"].median(),
    "Max_Alternative_Risk": ok_only["ok_tornado_alt_risk"].max()
})

hazard_rows.append({
    "State": "Washington",
    "Hazard": "Earthquake",
    "Mean_Alternative_Risk": wa_only["wa_earthquake_alt_risk"].mean(),
    "Median_Alternative_Risk": wa_only["wa_earthquake_alt_risk"].median(),
    "Max_Alternative_Risk": wa_only["wa_earthquake_alt_risk"].max()
})

hazard_rows.append({
    "State": "Washington",
    "Hazard": "Wildfire",
    "Mean_Alternative_Risk": wa_only["wa_wildfire_alt_risk"].mean(),
    "Median_Alternative_Risk": wa_only["wa_wildfire_alt_risk"].median(),
    "Max_Alternative_Risk": wa_only["wa_wildfire_alt_risk"].max()
})

summary_table_3 = pd.DataFrame(hazard_rows).round(4)
summary_table_3


# In[53]:


## Oklahoma composite alternative risk map
ok_map = risk_data[risk_data["STATE"] == "Oklahoma"].copy()
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ok_map.plot(
    column="alt_risk_composite",
    cmap="RdYlBu_r",
    linewidth=0,
    ax=ax,
    legend=True,
    legend_kwds={"label": "Alternative Composite Risk"}
)
ax.set_title("Oklahoma Composite Alternative Risk")
ax.axis("off")
plt.show()


# In[55]:


# Washington composite alternative risk map
wa_map = risk_data[risk_data["STATE"] == "Washington"].copy()

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
wa_map.plot(
    column="alt_risk_composite",
    cmap="RdYlBu_r",
    linewidth=0,
    ax=ax,
    legend=True, 
    legend_kwds={"label": "Alternative Composite Risk"}
)
ax.set_title("Washington Composite Alternative Risk")
ax.axis("off")
plt.show()


# In[56]:


## Oklahoma tornado alternative risk map
if "ok_tornado_alt_risk" in risk_data.columns:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ok_map.plot(
    column="ok_tornado_alt_risk",
    cmap="RdYlBu_r",
    linewidth=0,
    ax=ax,
    legend=True,
    legend_kwds={"label": "Oklahoma Tornado Alternative Risk"}
    )
    ax.set_title("Oklahoma Tornado Alternative Risk")
    ax.axis("off")
    plt.show()


# In[57]:


## Washington earthquake alternative risk map
if "wa_earthquake_alt_risk" in risk_data.columns:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    wa_map.plot(
    column="wa_earthquake_alt_risk",
    cmap="RdYlBu_r",
    linewidth=0,
    ax=ax,
    legend=True,
    legend_kwds={"label": "Washington Earthquake Alternative Risk"}
    )
    ax.set_title("Washington Earthquake Alternative Risk")
    ax.axis("off")
    plt.show()


# In[58]:


## Washington wildfire alternative risk map 
if "wa_wildfire_alt_risk" in risk_data.columns:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    wa_map.plot(
    column="wa_wildfire_alt_risk",
    cmap="RdYlBu_r",
    linewidth=0,
    ax=ax,
    legend=True,
    legend_kwds={"label": "Washington Wildfire Alternative Risk"}
    )
    ax.set_title("Washington Wildfire Alternative Risk")
    ax.axis("off")
    plt.show()

