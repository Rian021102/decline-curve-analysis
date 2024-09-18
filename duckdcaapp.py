import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import duckdb
import seaborn as sns

st.title("DECLINE CURVE ANALYSIS (DCA)")

#make connection to database
con=duckdb.connect('/Users/rianrachmanto/pypro/database/trialduckdb/trial.db')
QUERY = """
SELECT
    DATEPRD,
    NPD_WELL_BORE_NAME,
    BORE_OIL_VOL,
    BORE_GAS_VOL,
    BORE_WAT_VOL
FROM volve_well_test
WHERE
    BORE_OIL_VOL IS NOT NULL
    AND BORE_GAS_VOL IS NOT NULL
    AND BORE_WAT_VOL IS NOT NULL
ORDER BY
    NPD_WELL_BORE_NAME ASC, DATEPRD DESC
"""
st.set_option('deprecation.showPyplotGlobalUse', False)

def data_handler(QUERY):
    query_job=con.execute(QUERY)
    df=con.sql(QUERY).df()
    st.write(df.head())
    df_fil = df[(df['BORE_OIL_VOL'] > 0) & (df['BORE_GAS_VOL'] > 0) & (df['BORE_WAT_VOL'] > 0)].copy()
    df_fil.loc[:, 'DATEPRD'] = pd.to_datetime(df_fil['DATEPRD'])
    
    sns.set_theme(style="darkgrid")
    st.write(sns.relplot(
        data=df_fil,
        x="DATEPRD", y="BORE_OIL_VOL", col="NPD_WELL_BORE_NAME", hue="NPD_WELL_BORE_NAME",
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=2, height=3, aspect=1.5, legend=False
    ).fig)

    # Create a dataframe for monthly average
    df_monthly = df_fil.groupby(['NPD_WELL_BORE_NAME', pd.Grouper(key='DATEPRD', freq='M')]).mean()
    df_monthly = df_monthly.reset_index()
    df_monthly_24 = df_monthly[df_monthly['DATEPRD'] >= '2015-01-01']
    st.title("Monthly Average")
    sns.set_theme(style="darkgrid")
    st.write(sns.relplot(
        data=df_monthly_24,
        x="DATEPRD", y="BORE_OIL_VOL", col="NPD_WELL_BORE_NAME", hue="NPD_WELL_BORE_NAME",
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=2, height=3, aspect=1.5, legend=False
    ).fig)

    return df_monthly_24
df_monthly_24 = data_handler(QUERY)

# Add a "Forecast" button
if st.button("Forecast"):
    # Create an empty dictionary to store dataframes
    well_dataframes = {}
    
    # Iterate through unique well names and filter the data
    for well_name in df_monthly_24['NPD_WELL_BORE_NAME'].unique():
        well_df = df_monthly_24[df_monthly_24['NPD_WELL_BORE_NAME'] == well_name]
        well_dataframes[well_name] = well_df
    
    # Initialize forecast variables
    t_forecast_dict = {}
    q_forecast_dict = {}
    Qp_forecast_dict = {}

    # Iterate through unique well names and perform forecasting for each well
    for well_name, well_df in well_dataframes.items():
        st.write(f"Forecasting for Well: {well_name}")
        
        # Create a 't' array where t is DATEPRD
        t = well_df['DATEPRD'].values
        
        # Create a 'q' array where q is BORE_OIL_VOL
        q = well_df['BORE_OIL_VOL'].values
        
        # Subtract one datetime from another for 't'
        timedelta_t = [j - i for i, j in zip(t[:-1], t[1:])]
        timedelta_t = np.array(timedelta_t)
        timedelta_t = timedelta_t / np.timedelta64(1, 'D')  # Convert timedelta to days
        
        # Take cumulative sum over timedeltas for 't'
        t = np.cumsum(timedelta_t)
        t = np.append(0, t)
        t = t.astype(float)
        
        # Normalize 't' and 'q' data
        t_normalized = t / max(t)
        q_normalized = q / max(q)
        
        # Function for exponential decline
        def exponential(t, qi, di):
            return qi * np.exp(-di * t)
        
        # Fit the exponential decline model to the normalized data
        popt, pcov = curve_fit(exponential, t_normalized, q_normalized)
        qi, di = popt
        
        # Check if di is <= 0.0, if so, skip this well
        if di <= 0.0:
            print(f'Skipping well {well_name} due to di <= 0.0')
            continue
        
        # De-normalize qi and di
        qi = qi * max(q)
        di = di / max(t)
        
        # Initialize forecast variables
        t_forecast = []
        q_forecast = []
        Qp_forecast = []
        
        # Initial values
        t_current = 0
        q_current = exponential(t_current, qi, di)
        
        # Function to calculate cumulative production
        def cumpro(q_forecast, qi, di):
            return (qi - q_forecast) / di
        
        Qp_current = cumpro(q_current, qi, di)
        
        # Start forecasting until q_forecast reaches 25
        while q_current >= 25:
            t_forecast.append(t_current)
            q_forecast.append(q_current)
            Qp_forecast.append(Qp_current)
            
            # Increment time step
            t_current += 1
            q_current = exponential(t_current, qi, di)
            Qp_current = cumpro(q_current, qi, di)
        
        # Convert lists to numpy arrays for convenience
        t_forecast = np.array(t_forecast)
        q_forecast = np.array(q_forecast)
        Qp_forecast = np.array(Qp_forecast)
        
        # Display results in Streamlit
        st.write('Final Rate:', np.round(q_forecast[-1], 3), 'BOPD')
        st.write('Final Cumulative Production:', np.round(Qp_forecast[-1], 2), 'BBL OIL')
        
        # Plot the results using Matplotlib and display them in Streamlit
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(t, q, '.', color='red', label='Production Data')
        plt.plot(t_forecast, q_forecast, label='Forecast')
        plt.title('Oil Production Rate Result of DCA', size=13, pad=15)
        plt.xlabel('Days')
        plt.ylabel('Rate (BBL OIL/d)')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(t_forecast, Qp_forecast)
        plt.title('OIL Cumulative Production Result of DCA', size=13, pad=15)
        plt.xlabel('Days')
        plt.ylabel('Production (BBL OIL)')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        
        # Display the Matplotlib figure in Streamlit
        st.pyplot()
