import streamlit as st
import pandas as pd
import requests

def apply_filters(df, filters):
    filtered_df = df.copy()
    for column, filter_params in filters.items():
        filter_type = filter_params.get('type')
        value = filter_params.get('value')

        if filter_type and value is not None:
            if filter_type == "List":
                filtered_df = filtered_df[filtered_df[column].isin(value)]
            elif pd.api.types.is_string_dtype(df[column]):
                if filter_type == "Contains":
                    filtered_df = filtered_df[filtered_df[column].str.contains(value, na=False)]
                elif filter_type == "Does not contain":
                    filtered_df = filtered_df[~filtered_df[column].str.contains(value, na=False)]
                elif filter_type == "Starts with":
                    filtered_df = filtered_df[filtered_df[column].str.startswith(value, na=False)]
                elif filter_type == "Does not start with":
                    filtered_df = filtered_df[~filtered_df[column].str.startswith(value, na=False)]
                elif filter_type == "Ends with":
                    filtered_df = filtered_df[filtered_df[column].str.endswith(value, na=False)]
                elif filter_type == "Does not end with":
                    filtered_df = filtered_df[~filtered_df[column].str.endswith(value, na=False)]
                elif filter_type == "Equals":
                    filtered_df = filtered_df[filtered_df[column] == value]
                elif filter_type == "Not equals":
                    filtered_df = filtered_df[filtered_df[column] != value]
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                if filter_type == "Date Range":
                    start_date, end_date = value
                    filtered_df = filtered_df[(df[column] >= pd.to_datetime(start_date)) & (df[column] <= pd.to_datetime(end_date))]
            else:
                if filter_type == "Greater than":
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif filter_type == "Greater than or equal":
                    filtered_df = filtered_df[filtered_df[column] >= value]
                elif filter_type == "Less than":
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif filter_type == "Less than or equal":
                    filtered_df = filtered_df[filtered_df[column] <= value]
                elif filter_type == "Range":
                    min_val, max_val = value
                    filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]
                elif filter_type == "Top 10":
                    filtered_df = filtered_df.nlargest(10, column)
                elif filter_type == "Above average":
                    filtered_df = filtered_df[filtered_df[column] > filtered_df[column].mean()]
                elif filter_type == "Below average":
                    filtered_df = filtered_df[filtered_df[column] < filtered_df[column].mean()]

    return filtered_df

def filter_options(df, column):
    filter_type = st.selectbox(
        f"Select filter type for {column}",
        ["None", "List", "Contains", "Does not contain", "Starts with", "Does not start with", 
         "Ends with", "Does not end with", "Equals", "Not equals"] if pd.api.types.is_string_dtype(df[column]) else
        ["None", "List", "Greater than", "Greater than or equal", "Less than", 
         "Less than or equal", "Range", "Top 10", "Above average", "Below average"] if not pd.api.types.is_datetime64_any_dtype(df[column]) else
        ["None", "List", "Date Range"],
        key=f"{column}_filter_type"
    )
    
    value = None
    if filter_type == "List":
        value = st.multiselect(f"Values for {column}", df[column].unique(), key=f"{column}_list")
    elif pd.api.types.is_string_dtype(df[column]):
        value = st.text_input(f"Value for {column}", key=f"{column}_value")
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        if filter_type == "Date Range":
            value = st.date_input(
                f"Select date range for {column}",
                value=(df[column].min(), df[column].max()),
                key=f"{column}_date_range"
            )
    else:
        if filter_type in ["Greater than", "Greater than or equal", "Less than", "Less than or equal"]:
            value = st.number_input(f"Value for {column}", value=float(df[column].min()), key=f"{column}_num_value")
        elif filter_type == "Range":
            value = st.slider(
                f"Range for {column}", float(df[column].min()), float(df[column].max()), 
                (float(df[column].min()), float(df[column].max())),
                key=f"{column}_range"
            )

    # Update filters dynamically
    st.session_state.filters[column] = {'type': filter_type, 'value': value}

def interaction_filters_v1(df_forecast):
    if 'filters' not in st.session_state:
        st.session_state.filters = {col: {'type': None, 'value': None} for col in df_forecast.columns}
    if 'current_filter' not in st.session_state:
        st.session_state.current_filter = None

    # st.write("### Data Table")

    with st.container():
        col = st.columns(2)
        with col[0]:
            if st.button("Filter"):
                st.session_state.current_filter = "sidebar"
        with col[1]:
            if st.button("Reset Filters"):
                st.session_state.filters = {col: {'type': None, 'value': None} for col in df_forecast.columns}
                st.session_state.current_filter = None
        

    if st.session_state.current_filter == "sidebar":
        with st.sidebar.expander("Filter Options"):
            st.write("### Filter Options")
            selected_columns = st.multiselect("Select columns to filter", options=df_forecast.columns)

            # Remove filters for columns not selected
            for column in df_forecast.columns:
                if column not in selected_columns:
                    st.session_state.filters[column] = {'type': None, 'value': None}

            for column in selected_columns:
                filter_options(df_forecast, column)

    df_forecast_filtered = apply_filters(df_forecast, st.session_state.filters)
    st.write(df_forecast_filtered)

    return df_forecast_filtered 


if __name__ == "__main__":
    # Load data from API
    url = 'https://data.weather.gov.hk/weatherAPI/opendata/weather.php?dataType=fnd&lang=en'

    def get_data(url):
        response = requests.get(url)
        data = response.json()
        return data

    data = get_data(url)

    def data_weather_forecast(data):
        df = pd.DataFrame(data['weatherForecast'])
        df['forecastDate'] = pd.to_datetime(df['forecastDate'])
        df['week'] = df['forecastDate'].dt.day_name()
        return df

    df_forecast = data_weather_forecast(data)

    interaction_filters_v1(df_forecast)