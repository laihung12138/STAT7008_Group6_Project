#!/usr/bin/env python
# coding: utf-8
# 3.0.py

# In[ ]:

# In[1]:


import streamlit as st
import pandas as pd
import io
import os
import numpy as np
import tempfile
import altair as alt
import DataProcessor
import plot_update
import FurtherProcessor
from interaction_export_button import DataExporter
from interaction_filters_v1 import interaction_filters_v1
import get_location_lat_lon


# In[ ]:


class Dashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Dashboard",
            page_icon="🏂",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.init_session_state()
        self.app = plot_update.DataApp()  # Create a DataApp instance
        self.layouts = {
            "layout1": "Three-column layout",
            "layout2": "Two-column layout"
        }
    def init_session_state(self):
        if "pages" not in st.session_state:
            st.session_state.pages = ["dashboard1"]
        if "layouts" not in st.session_state:
            st.session_state.layouts = {"dashboard1": "layout1"}
        if "charts" not in st.session_state:
            st.session_state.charts = {"dashboard1": []}
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = {"dashboard1": {"file": None, "processed_data": None}}
    def manage_pages(self):
        with st.sidebar:
            with st.expander("Page Management"):
                new_page_name = st.text_input("New Dashboard Name")
                selected_layout = st.selectbox("Select Layout", self.layouts.keys())
                if st.button("Save Dashboard"):
                    if new_page_name not in st.session_state.pages:
                        st.session_state.pages.append(new_page_name)
                        st.session_state.layouts[new_page_name] = selected_layout
                        st.session_state.charts[new_page_name] = []
                        st.session_state.uploaded_files[new_page_name] = {"file": None, "processed_data": None}
#                         st.session_state.uploaded_files[new_page_name] = None  # Initialize uploaded_files for the new page
                        st.success("Saved successfully")
                    elif not new_page_name:
                        st.error("Please enter a new page name")
                    else:
                        st.error("The page already exists")
                selected_page = st.selectbox("Select Dashboard", st.session_state.pages)
                if st.button("Delete Current Page"):
                    if selected_page != "dashboard1":
                        st.session_state.pages.remove(selected_page)
                        del st.session_state.layouts[selected_page]
                        del st.session_state.charts[selected_page]
                        del st.session_state.uploaded_files[selected_page]
                        st.success(f"Page {selected_page} has been deleted")
                    else:
                        st.error("The default page cannot be deleted")
        return selected_page



    
    def display_data(self, selected_page):
        with st.expander("Data"):
            uploaded_file = st.file_uploader(f"Upload data file for {selected_page}", type=["csv", "xlsx"])
            
            
            # Save uploaded file to session_state
            if uploaded_file is not None:
                # Clear processed_data when new file is uploaded
                st.session_state.uploaded_files[selected_page] = {"file": uploaded_file, "processed_data": None,"preprocessed_data":None,"charts_customer":None}

            # Ensure the file exists before proceeding
            if selected_page in st.session_state.uploaded_files and st.session_state.uploaded_files[selected_page]["file"]:
                try:
                    # Always load the original file when 'Process' is clicked again
                    uploaded_file = st.session_state.uploaded_files[selected_page]["file"]
                    if uploaded_file.name.endswith('.csv'):
                        file_data = uploaded_file.getvalue()
                        df = pd.read_csv(io.StringIO(file_data.decode('utf-8')))
                    elif uploaded_file.name.endswith('.xlsx'):
                        file_data = uploaded_file.getvalue()
                        df = pd.read_excel(io.BytesIO(file_data))
                    else:
                        st.error("Unsupported file format.")
                        return  # Exit the function if unsupported file format is detected
                    if df.empty:
                        st.error("The uploaded file is empty.")
                        return  # Exit the function if the file is empty
                    if len(df.columns) < 2:
                        st.error("Data must have at least 2 columns.")
                        return  # Exit the function if insufficient columns
                    
                    file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    self.app.file_path = file_path
                    self.app.file_type = os.path.splitext(file_path)[1].lower()
                    if self.app.load_data() is not None:
                        self.app.render_monitoring_controls()
                        
                    # Save the loaded data for future processing
                    if st.session_state.uploaded_files[selected_page].get("processed_data") is None:
                        st.session_state.uploaded_files[selected_page]["processed_data"] = df
                    
                    # Save the DataFrame to session_state
                    st.session_state.df = df
                    
                    # Function to determine if a column is likely a date
                    def is_potential_date(series):
                        # Check if the column is a string or object type
                        if series.dtype == 'object':
                            # Check if the column contains date-like strings
                            try:
                                pd.to_datetime(series, errors='raise')  # Try to convert without coercion
                                return True
                            except Exception:
                                return False
                        return False

                    # Iterate over the columns and convert only the likely date columns
                    for col in df.columns:
                        if is_potential_date(df[col]):
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime
                            except Exception:
                                pass  # Ignore any conversion errors
                    # Ensure all potential datetime columns are converted to datetime64 type

                    # Update self.app.data and column classifications
                    self.app.data = df
                    self.app.numeric_cols = self.app.data.select_dtypes(include=['int64', 'float64']).columns
                    self.app.datetime_cols = self.app.data.select_dtypes(include=['datetime64']).columns
                    self.app.categorical_cols = self.app.data.select_dtypes(include=['object', 'category']).columns
    
                    
                    # # Add export button
                    # with st.sidebar:
                    #     if st.sidebar.button("Export Data"):
                    #         if st.session_state.df is not None:
                    #             exporter = DataExporter(st.session_state.df)
                    #             exporter.display_options()
                    #         else:
                    #             st.error("No data to export")

                    with st.sidebar.expander("Export Data"):
                        st.subheader("Export Data")
                        if st.session_state.uploaded_files.get(selected_page):
                            df_export = st.session_state.uploaded_files[selected_page]["processed_data"]
                            if df_export is not None:
                                exporter = DataExporter(df_export)
                                exporter.display_options()
                            else:
                                st.error("No processed data available to export.")


                    # Display Data Overview
                    st.subheader("Data Information")
                    with st.container():
                        info = {
                            "Total Rows": self.app.data.shape[0],
                            "Total Columns": self.app.data.shape[1],
                            "Numeric Columns": len(self.app.numeric_cols),
                            "Categorical Columns": len(self.app.categorical_cols),
                            "DateTime Columns": len(self.app.datetime_cols),
                            "Missing Values": self.app.data.isnull().sum().sum(),
                            "Memory Usage": f"{self.app.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                            "Duplicated Rows": self.app.data.duplicated().sum(),
                        }
                        info_items = list(info.items())
                        mid_point = len(info_items) // 2
                        col = st.columns((4, 1, 1))
                        with col[0]:
                            st.write("Data Preview:")
                            st.dataframe(self.app.data.head())  # Showing the full dataset as a preview
                        with col[1]:
                            for key, value in info_items[:mid_point]:
                                st.metric(label=key, value=value)
                        with col[2]:
                            for key, value in info_items[mid_point:]:
                                st.metric(label=key, value=value)
                    # Data Preprocessing Options
                    st.subheader("Data Preprocessing")
                    with st.container():
                        col = st.columns(3)
                        with col[0]:
                            st.write("For numerical variables")
                            method_number = st.selectbox("Handle Missing Values", ["mean", "median", "mode", "interpolate","forward fill","backward fill","delete"])
                        with col[1]:
                            st.write("For categorical variables")
                            method_object = st.selectbox("Handle Missing Values", ["mode", "forward fill", "backward fill","delete"])
                        with col[2]:
                            st.write("Choose columns you want to handle")
                            normalization_object = st.selectbox("Normalization", options=self.app.data.columns)
                    with st.container():
                        processor = DataProcessor.DataProcessor(self.app.data)# Apply preprocessing based on the selected options
                        if st.session_state.uploaded_files[selected_page].get("preprocessed_data") is None:
                            st.session_state.uploaded_files[selected_page]["preprocessed_data"] = processor.df
                        col = st.columns(2)
                        with col[0]:
                            if st.button("Preprocess"):
                            
                                
                                #processor.df = processor.handle_exception_datatype() #bug
                                processor.df = processor.handle_missing_values(method_number, method_object)
                                processor.df = processor.handle_outliers()
                                # doesn't use standardize_column_format()                               
                                # Save processed data back to session state
                                st.session_state.uploaded_files[selected_page]["processed_data"] = processor.df
                                st.session_state.uploaded_files[selected_page]["preprocessed_data"] = processor.df
                                self.app.data = processor.df
                        with col[1]:
                            if st.button("Normalize"):
                                further_processor = FurtherProcessor.FurtherProcessor(st.session_state.uploaded_files[selected_page]["preprocessed_data"])
                                further_processor.normalize_column(normalization_object)
                                
                                st.session_state.uploaded_files[selected_page]["processed_data"] = further_processor.data
                                st.session_state.uploaded_files[selected_page]["preprocessed_data"] = further_processor.data
                                self.app.data = further_processor.data
                        if st.session_state.uploaded_files[selected_page].get("processed_data") is not None:
                            st.write("Processed Data:")
                            st.dataframe(st.session_state.uploaded_files[selected_page]["preprocessed_data"].head())
        
        
                    
                    st.subheader("Columns Information")
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write("Numeric Columns:")
                            numeric_df = pd.DataFrame(self.app.numeric_cols, columns=["Columns"])
                            st.dataframe(numeric_df.transpose())
                        with col2:
                            st.write("Categorical Columns:")
                            categorical_df = pd.DataFrame(self.app.categorical_cols, columns=["Columns"])
                            st.dataframe(categorical_df.transpose())
                        with col3:
                            st.write("DateTime Columns:")
                            datetime_df = pd.DataFrame(self.app.datetime_cols, columns=["Columns"])
                            st.dataframe(datetime_df.transpose())
                            
                    with st.container():
                        st.subheader("Feature Engineering:")
                        # name_columns = st.session_state.uploaded_files[selected_page]["processed_data"].columns
                        name_columns = st.session_state.uploaded_files[selected_page]["processed_data"].columns.tolist()
                        selected_columns = []
                        furtherProcessor = FurtherProcessor.FurtherProcessor(st.session_state.uploaded_files[selected_page]["processed_data"])
                    with st.container():
                        col = st.columns(4)
                        with col[0]:
                            selected_column1 = st.selectbox("Select column to operate", options=name_columns,key="column1")
                        with col[1]:
                            operation = st.selectbox("Select operation", options=["+", "-", "×","÷"])
                            Operation = []
                            if operation == "+":
                                Operation = "add"
                            if operation == "-":
                                Operation = "subtract"
                            if operation == "×":
                                Operation = "multiply"
                            if operation == "÷":
                                Operation = "divide"
                        with col[2]:
                            selected_column2 = st.selectbox("Select column to operate", options=name_columns,key="column2")
                        with col[3]:
                            new_name_input = st.text_input("New Column Name", key="new_name")
                        
                        apply_button = st.button("Apply")
                        if apply_button:
                            selected_columns.append(selected_column1)
                            selected_columns.append(selected_column2)
                            result_operation = furtherProcessor.feature_engineering(selected_columns, Operation,new_name_input)
                            
                            self.app.data = result_operation
                            st.session_state.uploaded_files[selected_page]["processed_data"] = self.app.data
                            st.write("Processed Data:")
                            st.dataframe(result_operation.head()) 
                        
            
                    # Display the processed data
                    if "processed_data" in st.session_state.uploaded_files[selected_page]:
                        st.subheader("Full Dataset")  # Add subtitle here
                        # Filter the data by calling interaction_filters_v1
                        # interaction_filters_v1(st.session_state.uploaded_files[selected_page]["processed_data"])
                        # st.dataframe(st.session_state.uploaded_files[selected_page]["processed_data"])
                    
                        df_filter = interaction_filters_v1(st.session_state.uploaded_files[selected_page]["processed_data"])
                        st.session_state.uploaded_files[selected_page]["preprocessed_data"] = df_filter
                        self.app.data = df_filter
                                            
                    # with st.container():
                    #     st.subheader("Filter Data:")  # Consider whether to link this with the subsequent chart generation
                    #     # Get the filtered data
                except Exception as e:
                    st.error(f"File read error: {e}")
                    return None
            else:
                # Handle the case when no file is uploaded
                st.info("Please upload a file to proceed.")
                return None
    
    def display_graphs(self, selected_page):
        with st.expander("Univariate Analysis Chart"):
            if st.session_state.uploaded_files.get(selected_page) is not None: 
                st.subheader("Univariate Analysis Chart")
                analysis_type = st.selectbox(
                   "Select Analysis Type",
                   ["Numerical Analysis", "Categorical Analysis", 
                    "Time Series Analysis"]
               )
                 
                if analysis_type == "Numerical Analysis":
                    st.subheader("Numerical Analysis")
                    if len(self.app.numeric_cols) == 0:
                        st.warning("No numerical columns found in the dataset")
                    else:
                        selected_num_col = st.selectbox("Select Numerical Column", self.app.numeric_cols)
                        self.app.plot_num(selected_num_col)
                if analysis_type == "Categorical Analysis":
                    st.subheader("Categorical Analysis")
                    if len(self.app.categorical_cols) == 0:
                        st.warning("No categorical columns found in the dataset")
                    else:
                        selected_cat_col = st.selectbox("Select Categorical Column", self.app.categorical_cols)
                        self.app.plot_cat(selected_cat_col)
                if analysis_type == "Time Series Analysis":
                    self.app.render_time_series_section()
                
        with st.expander("Bivariate Analysis Chart"):
            
            if st.session_state.uploaded_files.get(selected_page) is not None:  # Check if the key exists
                layout = st.session_state.layouts[selected_page]
                if st.button("Add Graph"):
                    st.session_state.charts[selected_page].append({})
                def render_charts(layout, charts):
                    if layout == "layout1":
                        col1, col2 = st.columns(2)
                        for i, chart_info in enumerate(charts):
                            if i % 2 == 0:
                                with col1:
                                    self.render_chart_Bivariate(i, selected_page, chart_info)
                            else:
                                with col2:
                                    self.render_chart_Bivariate(i, selected_page, chart_info)
                    elif layout == "layout2":
                        col1, col2, col3 = st.columns(3)
                        for i, chart_info in enumerate(charts):
                            if i % 3 == 0:
                                with col1:
                                    self.render_chart_Bivariate(i, selected_page, chart_info)
                            elif i % 3 == 1:
                                with col2:
                                    self.render_chart_Bivariate(i, selected_page, chart_info)
                            else:
                                with col3:
                                    self.render_chart_Bivariate(i, selected_page, chart_info)
                layout = st.session_state.layouts.get(selected_page, "layout1")
                render_charts(layout, st.session_state.charts[selected_page])
                
        with st.expander("Multivariate Analysis"):     
            if st.session_state.uploaded_files.get(selected_page) is not None:  # Check if the key exists
                layout = st.session_state.layouts[selected_page]
                if st.button("Add Graph",key="add_multivariate"):
                    st.session_state.charts[selected_page].append({})
                def render_charts(layout, charts):
                    if layout == "layout1":
                        col1, col2 = st.columns(2)
                        for i, chart_info in enumerate(charts):
                            if i % 2 == 0:
                                with col1:
                                    self.render_chart_Multivariate(i, selected_page, chart_info)
                            else:
                                with col2:
                                    self.render_chart_Multivariate(i, selected_page, chart_info)
                    elif layout == "layout2":
                        col1, col2, col3 = st.columns(3)
                        for i, chart_info in enumerate(charts):
                            if i % 3 == 0:
                                with col1:
                                    self.render_chart_Multivariate(i, selected_page, chart_info)
                            elif i % 3 == 1:
                                with col2:
                                    self.render_chart_Multivariate(i, selected_page, chart_info)
                            else:
                                with col3:
                                    self.render_chart_Multivariate(i, selected_page, chart_info)
                layout = st.session_state.layouts.get(selected_page, "layout1")
                render_charts(layout, st.session_state.charts[selected_page])
        
        with st.expander("Crossfiltering graph"):
            processed_data = st.session_state.uploaded_files[selected_page].get("processed_data", None)   # Check if 'processed_data' is not None
            if processed_data is not None:
                columnX = st.selectbox("Select X", st.session_state.uploaded_files[selected_page]["processed_data"].columns, disabled=(processed_data is None))
                columnY = st.selectbox("Select Y", st.session_state.uploaded_files[selected_page]["processed_data"].columns, disabled=(processed_data is None))
                columnCat = st.selectbox("Select Category", st.session_state.uploaded_files[selected_page]["processed_data"].columns, disabled=(processed_data is None))
                plot_type = st.multiselect("Plot Type", ["Bar Chart", "Pie Chart", "Line Plot"],max_selections=2)
                data=st.session_state.uploaded_files[selected_page]["processed_data"]
                colors = ["red","green","blue","purple","pink","yellow","orange","green"]
                if "Bar Chart" in plot_type and "Pie Chart" in plot_type:
                    Cat_select = alt.selection_single(fields=[columnCat], empty="all")
                    pie = (
                        (
                            alt.Chart(data)
                            .mark_arc(innerRadius=50)
                            .encode(
                                theta=alt.Theta(
                                    columnY,
                                    type="quantitative",
                                    aggregate="sum",
                                    title=columnY,
                                ),
                                color=alt.Color(
                                    field=columnCat,
                                    type="nominal",
                                    scale=alt.Scale(domain=data[columnCat], range=colors),
                                    title=columnCat,
                                ),
                                opacity=alt.condition(Cat_select, alt.value(1), alt.value(0.25)),
                            )
                        )
                        .transform_filter(Cat_select)
                        .add_selection(Cat_select)
                        .properties(title=columnY)
                    )
    
                    #stacked bar chart
                    stackedbar = (
                        (
                            alt.Chart(data)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    columnX,
                                ),
                                y=alt.Y(
                                    field=columnY,
                                    type="quantitative",
                                    aggregate="sum",
                                    title=columnY,
                                ),
                                color=alt.Color(
                                    columnCat,
                                    type="nominal",
                                    title=columnCat,
                                    scale=alt.Scale(domain=data[columnCat], range=colors),
                                    legend=alt.Legend(
                                        direction="vertical",
                                        symbolType="triangle-left",
                                        tickCount=4,
                                    ),
                                ),
                            )
                        )
                        .transform_filter(Cat_select)
                        .add_selection(Cat_select)
                        .properties(width=700, title=columnY)
                    )
                    full_chart =  pie | stackedbar
                    st.altair_chart(full_chart,use_container_width=True)
    
    
                if "Bar Chart" in plot_type and "Line Plot" in plot_type :
                    Cat_select = alt.selection_single(fields=[columnCat], empty="all")
                    #line Chart
                    line_chart = (
                        (
                            alt.Chart(data)
                            .mark_line()
                            .encode(
                                x=alt.X(
                                    columnX,
                                ),
                                y=alt.Y(
                                    field=columnY,
                                    type="quantitative",
                                    aggregate="sum",
                                    title=columnY,
                                ),
                                color=alt.Color(
                                    columnCat,
                                    type="nominal",
                                    title=columnCat,
                                    scale=alt.Scale(domain=data[columnCat], range=colors),
                                    legend=alt.Legend(
                                        direction="vertical",
                                        symbolType="triangle-left",
                                        tickCount=4,
                                    ),
                                ),
                            )
                        )
                        .transform_filter(Cat_select)
                        .add_selection(Cat_select)
                        .properties(width=700, height=550, title=columnY)
                    )
                    #stacked bar chart
                    stackedbar = (
                        (
                            alt.Chart(data)
                            .mark_bar()
                            .encode(
                                x=alt.X(
                                    columnX,
                                ),
                                y=alt.Y(
                                    field=columnY,
                                    type="quantitative",
                                    aggregate="sum",
                                    title=columnY,
                                ),
                                color=alt.Color(
                                    columnCat,
                                    type="nominal",
                                    title=columnCat,
                                    scale=alt.Scale(domain=data[columnCat], range=colors),
                                    legend=alt.Legend(
                                        direction="vertical",
                                        symbolType="triangle-left",
                                        tickCount=4,
                                    ),
                                ),
                            )
                        )
                        .transform_filter(Cat_select)
                        .add_selection(Cat_select)
                        .properties(width=700, title=columnY)
                    )
    
                    full_chart =  line_chart | stackedbar
                    st.altair_chart(full_chart,use_container_width=True)
    
    
    
                if "Pie Chart" in plot_type and "Line Plot" in plot_type:
                    Cat_select = alt.selection_single(fields=[columnCat], empty="all")
                    #line Chart
                    line_chart = (
                        (
                            alt.Chart(data)
                            .mark_line()
                            .encode(
                                x=alt.X(
                                    columnX,
                                ),
                                y=alt.Y(
                                    field=columnY,
                                    type="quantitative",
                                    aggregate="sum",
                                    title=columnY,
                                ),
                                color=alt.Color(
                                    columnCat,
                                    type="nominal",
                                    title=columnCat,
                                    scale=alt.Scale(domain=data[columnCat], range=colors),
                                    legend=alt.Legend(
                                        direction="vertical",
                                        symbolType="triangle-left",
                                        tickCount=4,
                                    ),
                                ),
                            )
                        )
                        .transform_filter(Cat_select)
                        .add_selection(Cat_select)
                        .properties(width=700, height=550, title=columnY)
                    )
                    pie = (
                        (
                            alt.Chart(data)
                            .mark_arc(innerRadius=50)
                            .encode(
                                theta=alt.Theta(
                                    columnY,
                                    type="quantitative",
                                    aggregate="sum",
                                    title=columnY,
                                ),
                                color=alt.Color(
                                    field=columnCat,
                                    type="nominal",
                                    scale=alt.Scale(domain=data[columnCat], range=colors),
                                    title=columnCat,
                                ),
                                opacity=alt.condition(Cat_select, alt.value(1), alt.value(0.25)),
                            )
                        )
                        .transform_filter(Cat_select)
                        .add_selection(Cat_select)
                        .properties(title=columnY)
                    )
    
                    full_chart =  line_chart | pie
                    st.altair_chart(full_chart,use_container_width=True)

    def render_chart_Bivariate(self, i, selected_page, chart_info):
        bivariate_data_type = st.selectbox("Bivariate Data Type", ["Numerical vs Numerical", "Categorical vs Numerical", "Categorical vs Categorical"],key=f"bivariate_data_type_{selected_page}_{i}")
        if bivariate_data_type == "Numerical vs Numerical":
            x_col = st.selectbox("Select X Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"x_col_{selected_page}_{i}_b")
            y_col = st.selectbox("Select Y Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"y_col_{selected_page}_{i}_b")
            plot_type = st.selectbox("Plot Type", ["2D Density Plot", "Scatter Plot", "Line Plot", "Joint Plot", "Marginal Plot"],key=f"plot_type_{selected_page}_{i}_b")
            if plot_type == "2D Density Plot":
                plot_update.plot_2d_density(st.session_state.uploaded_files[selected_page]["processed_data"], x_col, y_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Scatter Plot":
                plot_update.plot_scatter(st.session_state.uploaded_files[selected_page]["processed_data"], x_col, y_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Line Plot":
                plot_update.plot_line(st.session_state.uploaded_files[selected_page]["processed_data"], x_col, y_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Joint Plot":
                plot_update.plot_joint(st.session_state.uploaded_files[selected_page]["processed_data"], x_col, y_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Marginal Plot":
                plot_update.plot_marginal(st.session_state.uploaded_files[selected_page]["processed_data"], x_col, y_col, key_suffix=f"{selected_page}_{i}")
        

        if bivariate_data_type == "Categorical vs Numerical":
            category_col = st.selectbox("Select Categorical Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"category_col_b_{i}")
            value_col = st.selectbox("Select Numerical Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"value_col_b_{i}")
            plot_type = st.selectbox("Plot Type", ["Box Plot", "Violin Plot", "Bar Plot", "Sunburst Chart", "Map Plot"],key=f"plot_type_b_{selected_page}_{i}")
            if plot_type == "Box Plot":
                plot_update.box_plot(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, value_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Violin Plot":
                plot_update.violin_plot(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, value_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Bar Plot":
                plot_update.bar_plot(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, value_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Sunburst Chart":
                plot_update.sunburst_chart(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, value_col, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Map Plot":
                # Read key from txt file
                google_map_api_key = open("google_map_api_key.txt", "r").read()
                plot_update.plot_map(st.session_state.uploaded_files[selected_page]["processed_data"], get_location_lat_lon.get_lat_lon_columns(st.session_state.uploaded_files[selected_page]["processed_data"][category_col], google_map_api_key), value_col)

        
        if bivariate_data_type == "Categorical vs Categorical":
            category1 = st.selectbox("Select First Categorical Column",st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"cat_cat_1_{i}")
            category2 = st.selectbox("Select Second Categorical Column",st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"cat_cat_2_{i}")
            plot_type = st.selectbox("Plot Type", ["Mosaic Plot", "Heatmap"],key=f"cat_cat_3_{i}")
            if plot_type == "Mosaic Plot":
                plot_update.mosaic_plot(st.session_state.uploaded_files[selected_page]["processed_data"], category1, category2, key_suffix=f"{selected_page}_{i}")
            elif plot_type == "Heatmap":
                plot_update.heatmap(st.session_state.uploaded_files[selected_page]["processed_data"], category1, category2, key_suffix=f"{selected_page}_{i}")
    
    def render_chart_Multivariate(self, i, selected_page, chart_info):
        with st.container():
            
            
            multivariate_data_type = st.selectbox("Multivariate Data Type", ["Numerical Data", "Categorical Data","Time Series"],key=f"multivariate_data_type_{selected_page}_{i}")
            if multivariate_data_type == "Numerical Data":
                columns = st.multiselect(f"Select Columns,chart{i+1}", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"columns_{selected_page}_{i}_Multivariate")
                plot_type = st.selectbox("Plot Type", ["Pair Plot", "Heatmap", "Bubble Chart"],key=f"plot_type_{selected_page}_{i}_Multivariate")
                if plot_type == "Pair Plot":
                    plot_update.plot_pair(st.session_state.uploaded_files[selected_page]["processed_data"][columns], columns, key_suffix=f"{selected_page}_{i}")
                elif plot_type == "Heatmap":
                    plot_update.plot_heatmap(st.session_state.uploaded_files[selected_page]["processed_data"][columns], columns, key_suffix=f"{selected_page}_{i}")
                elif plot_type == "Bubble Chart":
                    size_col = st.selectbox("Select Size Column", st.session_state.uploaded_files[selected_page]["processed_data"][columns].select_dtypes(include=np.number).columns)
                    plot_update.plot_bubble(st.session_state.uploaded_files[selected_page]["processed_data"][columns], columns[0], columns[1], size_col, key_suffix=f"{selected_page}_{i}")
            
            if multivariate_data_type == "Categorical Data":
                with st.container():
                    col = st.columns(2)
                    with col[0]:
                        category_col = st.selectbox("Select Categorical Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"category_col_Multivariate_{i}")
                    with col[1]:
                        subcategory_col = st.selectbox("Select Subcategory Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"subcategory_col_Multivariate_{i}")
                with st.container():
                    col = st.columns(2)
                    with col[0]:
                        value_col = st.selectbox("Select Value Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"value_col_Multivariate_{i}")
                    with col[1]:
                        plot_type = st.selectbox("Plot Type", ["Stacked Bar Plot", "Grouped Bar Plot", "Grouped Violin Box", "Clustered Bar Chart"],key=f"plot_type_Multivariate_{i}")
                if plot_type == "Stacked Bar Plot":
                    plot_update.stacked_bar_plot(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, subcategory_col, value_col, key_suffix=f"{selected_page}_{i}")
                elif plot_type == "Grouped Bar Plot":
                    plot_update.grouped_bar_plot(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, subcategory_col, value_col, key_suffix=f"{selected_page}_{i}")
                elif plot_type == "Grouped Violin Box":
                    plot_type_box_violin = st.selectbox("Select Plot Type", ["Violin", "Box"],key=f"plot_type_box_violin_{i}")
                    plot_update.grouped_violin_box(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, subcategory_col, value_col, plot_type_box_violin, key_suffix=f"{selected_page}_{i}")
                elif plot_type == "Clustered Bar Chart":
                    plot_update.clustered_bar_chart(st.session_state.uploaded_files[selected_page]["processed_data"], category_col, subcategory_col, value_col, key_suffix=f"{selected_page}_{i}")   
                    
            if multivariate_data_type == "Time Series":
                with st.container():
                    col = st.columns(3)
                    with col[0]:
                        date_col = st.selectbox("Select Date Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"date_col_Multivariate_{i}")
                    with col[1]:
                        value_col = st.multiselect("Select Value Columns", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"value_col_Multivariate_{i}")
                    with col[2]:
                        plot_type = st.selectbox("Plot Type", ["Candlestick Chart", "Time Series Correlation Matrix"], key=f"plot_type_Multivariate_{i}")
                if plot_type == "Candlestick Chart":
                    col = st.columns(4)
                    with col[0]:
                        open_col = st.selectbox("Open Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"pen_col_Multivariate_{i}")
                    with col[1]:
                        high_col = st.selectbox("High Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"high_col_Multivariate_{i}")
                    with col[2]:
                        low_col = st.selectbox("Low Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"low_col_Multivariate_{i}")
                    with col[3]:
                        close_col = st.selectbox("Close Column", st.session_state.uploaded_files[selected_page]["processed_data"].columns,key=f"close_col_Multivariate_{i}")
                    plot_update.candlestick_chart(st.session_state.uploaded_files[selected_page]["processed_data"], date_col, open_col, high_col, low_col, close_col, key_suffix=f"{selected_page}_{i}")
                if plot_type == "Time Series Correlation Matrix":
                    
                    plot_update.time_series_correlation_matrix(st.session_state.uploaded_files[selected_page]["processed_data"], value_col, key_suffix=f"{selected_page}_{i}")     

    
    def run(self):
        selected_page = self.manage_pages()
        self.display_data(selected_page)
        self.display_graphs(selected_page)


# In[ ]:


if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()

