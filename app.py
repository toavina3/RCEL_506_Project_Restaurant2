import ast
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ==============================================================================
# 1. PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Restaurant Processing Time Analytics",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⏱️ Restaurant Order Processing Time Analytics")
st.markdown(
    "Interactive dashboard analyzing how **Order Volume, Employee Count, Item Count, "
    "Modifiers, Order Type, Hour of Day, and Day of Week** impact order processing time all at once."
)

# ==============================================================================
# 2. DATA LOADING & PREPROCESSING (ADAPTED FROM ORIGINAL SCRIPT)
# ==============================================================================


def extract_counts(val):
    """Extract item count and modifier/customization count from lineItems JSON string or list."""
    if pd.isna(val):
        return 0, 0
    if isinstance(val, list):
        items = val
    elif isinstance(val, str):
        try:
            items = json.loads(val)
        except Exception:
            try:
                items = ast.literal_eval(val)
            except Exception:
                return 0, 0
    else:
        return 0, 0

    if not isinstance(items, list):
        return 0, 0

    items_count = len(items)
    custom_count = sum(
        len(item.get("modifications", {}).get("elements", []))
        for item in items
        if isinstance(item, dict)
    )
    return items_count, custom_count


@st.cache_data(show_spinner=False)
def load_and_preprocess_data(orders_url_or_path, shifts_url_or_path):
    # Load raw CSVs
    df1 = pd.read_csv(orders_url_or_path)
    df3 = pd.read_csv(shifts_url_or_path)

    # Convert timestamps to datetime and shift to Houston time (UTC-5)
    df1["createdTime"] = pd.to_datetime(
        df1["createdTime"], unit="ms"
    ) - pd.Timedelta(hours=5)
    df1["clientCreatedTime"] = pd.to_datetime(
        df1["clientCreatedTime"], unit="ms"
    ) - pd.Timedelta(hours=5)
    df1["modifiedTime"] = pd.to_datetime(
        df1["modifiedTime"], unit="ms"
    ) - pd.Timedelta(hours=5)

    df3["inTime"] = pd.to_datetime(df3["inTime"], unit="ms") - pd.Timedelta(
        hours=5
    )
    df3["outTime"] = pd.to_datetime(df3["outTime"], unit="ms") - pd.Timedelta(
        hours=5
    )

    # Merge orders with employee shifts
    df_combined = pd.merge(df1, df3, on="employee.id", how="left")

    # Strictly map order to active shift when createdTime falls between inTime and outTime
    df_combined = df_combined[
        (df_combined["createdTime"] >= df_combined["inTime"])
        & (df_combined["createdTime"] <= df_combined["outTime"])
    ].copy()

    # Select relevant columns
    cols_orders = [
        "id_x",
        "createdTime",
        "clientCreatedTime",
        "modifiedTime",
        "employee.id",
        "employee.name_x",
        "orderType.label",
        "lineItems.elements",
    ]
    cols_shifts = ["inTime", "outTime"]
    df_filtered = df_combined[cols_orders + cols_shifts].copy()

    # Derive date, hour, and processing time in minutes
    df_filtered["date"] = df_filtered["createdTime"].dt.date
    df_filtered["hour"] = df_filtered["createdTime"].dt.hour
    df_filtered["processing_time_min"] = (
        df_filtered["modifiedTime"] - df_filtered["createdTime"]
    ).dt.total_seconds() / 60.0

    # Exclude invalid or extreme outliers (e.g. negative time or > 120 min)
    df_filtered = df_filtered[
        (df_filtered["processing_time_min"] >= 0)
        & (df_filtered["processing_time_min"] <= 120)
    ]

    # Calculate employee count per date and hour
    emp_per_hour = (
        df_filtered.groupby(["date", "hour"])["employee.id"]
        .nunique()
        .reset_index()
        .rename(columns={"employee.id": "employee_count"})
    )
    df_filtered = pd.merge(
        df_filtered, emp_per_hour, on=["date", "hour"], how="left"
    )

    # Calculate order volume per date and hour
    vol_per_hour = (
        df_filtered.groupby(["date", "hour"])["id_x"]
        .nunique()
        .reset_index()
        .rename(columns={"id_x": "order_volume"})
    )
    df_filtered = pd.merge(
        df_filtered, vol_per_hour, on=["date", "hour"], how="left"
    )

    # Extract item count and modifier count from JSON
    counts = df_filtered["lineItems.elements"].apply(extract_counts)
    df_filtered["num_items"] = [c[0] for c in counts]
    df_filtered["num_modifiers"] = [c[1] for c in counts]

    # Day of week
    df_filtered["day_of_week"] = df_filtered["createdTime"].dt.day_name()

    # Fill any missing values in order type
    df_filtered["orderType.label"] = df_filtered["orderType.label"].fillna(
        "Unknown"
    )

    return df_filtered


# ==============================================================================
# 3. SIDEBAR - FILE PATHS & INTERACTIVE FILTERS
# ==============================================================================
st.sidebar.header("📁 Data Source Configuration")

orders_path = st.sidebar.text_input(
    "Orders CSV Path / Github Raw URL",
    value="orders.csv",
    help="Relative path in repo or direct raw GitHub URL",
)
shifts_path = st.sidebar.text_input(
    "Shifts CSV Path / Github Raw URL",
    value="shifts.csv",
    help="Relative path in repo or direct raw GitHub URL",
)

try:
    with st.spinner("Loading and processing data..."):
        df = load_and_preprocess_data(orders_path, shifts_path)
    st.sidebar.success(f"Data loaded: {len(df):,} orders")
except Exception as e:
    st.error(
        f"Error loading CSV files. Please check paths/URLs. Details: {str(e)}"
    )
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Dynamic Filters")

# Day of week filter
days_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
available_days = [d for d in days_order if d in df["day_of_week"].unique()]
selected_days = st.sidebar.multiselect(
    "Day of the Week", options=available_days, default=available_days
)

# Order type filter
order_types = sorted(df["orderType.label"].astype(str).unique().tolist())
selected_order_types = st.sidebar.multiselect(
    "Order Type", options=order_types, default=order_types
)

# Hour of Day Filter
min_hour, max_hour = int(df["hour"].min()), int(df["hour"].max())
selected_hours = st.sidebar.slider(
    "Hour of Day (0-23)", min_hour, max_hour, (min_hour, max_hour)
)

# Employee Count Filter
min_emp, max_emp = int(df["employee_count"].min()), int(
    df["employee_count"].max()
)
selected_emps = st.sidebar.slider(
    "Working Employees per Hour", min_emp, max_emp, (min_emp, max_emp)
)

# Order Volume Filter
min_vol, max_vol = int(df["order_volume"].min()), int(df["order_volume"].max())
selected_vol = st.sidebar.slider(
    "Hourly Order Volume", min_vol, max_vol, (min_vol, max_vol)
)

# Items & Modifiers Sliders
max_items_val = int(df["num_items"].max())
selected_items = st.sidebar.slider(
    "Number of Items", 0, max_items_val, (0, max_items_val)
)

max_mods_val = int(df["num_modifiers"].max())
selected_mods = st.sidebar.slider(
    "Number of Modifiers", 0, max_mods_val, (0, max_mods_val)
)

# Apply Filters
filtered_df = df[
    (df["day_of_week"].isin(selected_days))
    & (df["orderType.label"].isin(selected_order_types))
    & (df["hour"].between(selected_hours[0], selected_hours[1]))
    & (df["employee_count"].between(selected_emps[0], selected_emps[1]))
    & (df["order_volume"].between(selected_vol[0], selected_vol[1]))
    & (df["num_items"].between(selected_items[0], selected_items[1]))
    & (df["num_modifiers"].between(selected_mods[0], selected_mods[1]))
]

# ==============================================================================
# 4. DASHBOARD KPIS
# ==============================================================================
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Filtered Orders", f"{len(filtered_df):,}")
kpi2.metric(
    "Avg Processing Time",
    (
        f"{filtered_df['processing_time_min'].mean():.2f} min"
        if not filtered_df.empty
        else "N/A"
    ),
)
kpi3.metric(
    "Avg Items / Order",
    (
        f"{filtered_df['num_items'].mean():.2f}"
        if not filtered_df.empty
        else "N/A"
    ),
)
kpi4.metric(
    "Avg Modifiers / Order",
    (
        f"{filtered_df['num_modifiers'].mean():.2f}"
        if not filtered_df.empty
        else "N/A"
    ),
)
kpi5.metric(
    "Avg Staff / Hour",
    (
        f"{filtered_df['employee_count'].mean():.1f}"
        if not filtered_df.empty
        else "N/A"
    ),
)

st.markdown("---")

if filtered_df.empty:
    st.warning("No data matches the selected filter criteria.")
    st.stop()

# ==============================================================================
# 5. MULTI-VARIABLE ANALYTICS TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "🌐 All-in-One Multi-Variable Views",
        "📊 Pairwise Relationships & Heatmap",
        "🤖 Statistical Drivers & Predictor",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: ALL-IN-ONE MULTI-VARIABLE VISUALIZATIONS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader(
        "1. Parallel Coordinates Plot (Visualizing All 7 Drivers Simultaneously)"
    )
    st.caption(
        "Trace individual order paths across Day, Hour, Order Type, Items, Modifiers, Volume, Staff, and Processing Time."
    )

    p_df = filtered_df.copy()
    # Map categorical variables to numeric codes for Parallel Coordinates
    p_df["day_code"] = p_df["day_of_week"].map(
        {d: i for i, d in enumerate(days_order)}
    )
    p_df["type_code"] = p_df["orderType.label"].astype("category").cat.codes

    fig_parcoord = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=p_df["processing_time_min"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Processing (min)"),
            ),
            dimensions=[
                dict(
                    range=[0, 6],
                    tickvals=list(range(7)),
                    ticktext=[d[:3] for d in days_order],
                    label="Day of Week",
                    values=p_df["day_code"],
                ),
                dict(
                    range=[0, 23],
                    label="Hour of Day",
                    values=p_df["hour"],
                ),
                dict(
                    range=[0, p_df["type_code"].max() or 1],
                    label="Order Type",
                    values=p_df["type_code"],
                ),
                dict(
                    range=[0, p_df["num_items"].max() or 1],
                    label="Num Items",
                    values=p_df["num_items"],
                ),
                dict(
                    range=[0, p_df["num_modifiers"].max() or 1],
                    label="Num Modifiers",
                    values=p_df["num_modifiers"],
                ),
                dict(
                    range=[
                        p_df["order_volume"].min(),
                        p_df["order_volume"].max(),
                    ],
                    label="Order Volume",
                    values=p_df["order_volume"],
                ),
                dict(
                    range=[
                        p_df["employee_count"].min(),
                        p_df["employee_count"].max(),
                    ],
                    label="Employee Count",
                    values=p_df["employee_count"],
                ),
                dict(
                    range=[
                        p_df["processing_time_min"].min(),
                        p_df["processing_time_min"].max(),
                    ],
                    label="Processing Time (min)",
                    values=p_df["processing_time_min"],
                ),
            ],
        )
    )
    fig_parcoord.update_layout(height=450, margin=dict(l=60, r=60, t=50, b=40))
    st.plotly_chart(fig_parcoord, use_container_width=True)

    st.subheader("2. Interactive 5D Bubble Scatter Plot")
    st.caption(
        "Examine 5 dimensions at once: X-axis, Y-axis, Color, Size, and Facet Grid."
    )

    col_x, col_y, col_color, col_size, col_facet = st.columns(5)
    axis_opts = [
        "employee_count",
        "order_volume",
        "num_items",
        "num_modifiers",
        "hour",
        "processing_time_min",
    ]

    x_axis = col_x.selectbox("X-Axis", axis_opts, index=0)
    y_axis = col_y.selectbox("Y-Axis", axis_opts, index=5)
    color_axis = col_color.selectbox(
        "Color Code", ["orderType.label", "day_of_week"] + axis_opts, index=0
    )
    size_axis = col_size.selectbox(
        "Bubble Size", ["num_items", "num_modifiers", "order_volume"], index=0
    )
    facet_axis = col_facet.selectbox(
        "Facet Split", ["None", "day_of_week", "orderType.label"], index=0
    )

    scatter_kwargs = {
        "data_frame": filtered_df,
        "x": x_axis,
        "y": y_axis,
        "color": color_axis,
        "size": size_axis,
        "hover_data": [
            "createdTime",
            "employee.name_x",
            "num_items",
            "num_modifiers",
            "processing_time_min",
        ],
        "opacity": 0.7,
    }
    if facet_axis != "None":
        scatter_kwargs["facet_col"] = facet_axis

    fig_scatter = px.scatter(**scatter_kwargs)
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: PAIRWISE RELATIONSHIPS & HEATMAP
# ------------------------------------------------------------------------------
with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Correlation Matrix Heatmap")
        num_cols = [
            "processing_time_min",
            "order_volume",
            "employee_count",
            "num_items",
            "num_modifiers",
            "hour",
        ]
        corr = filtered_df[num_cols].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Correlation with Processing Time",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_right:
        st.subheader("Processing Time by Order Type")
        fig_type = px.box(
            filtered_df,
            x="orderType.label",
            y="processing_time_min",
            color="orderType.label",
            points=False,
            title="Processing Time Distribution Across Order Types",
        )
        st.plotly_chart(fig_type, use_container_width=True)

    st.subheader("Temporal Drivers: Hour of Day & Day of Week")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        hourly_summary = (
            filtered_df.groupby("hour")["processing_time_min"]
            .agg(["mean", "median"])
            .reset_index()
        )
        fig_hour = px.line(
            hourly_summary,
            x="hour",
            y=["mean", "median"],
            markers=True,
            title="Avg & Median Processing Time by Hour of Day",
            labels={
                "value": "Processing Time (min)",
                "variable": "Metric",
                "hour": "Hour",
            },
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with col_t2:
        day_summary = (
            filtered_df.groupby("day_of_week")["processing_time_min"]
            .mean()
            .reindex(available_days)
            .reset_index()
        )
        fig_day = px.bar(
            day_summary,
            x="day_of_week",
            y="processing_time_min",
            color="day_of_week",
            title="Average Processing Time by Day of Week",
        )
        st.plotly_chart(fig_day, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: STATISTICAL DRIVERS & WHAT-IF PREDICTOR
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Multivariate Machine Learning Model (Feature Importance)")
    st.caption(
        "Random Forest model measuring how much each factor impacts processing time."
    )

    # Encode categorical features for regression model
    df_ml = filtered_df.copy()
    df_ml = pd.get_dummies(
        df_ml, columns=["orderType.label", "day_of_week"], drop_first=True
    )

    features = [
        c
        for c in df_ml.columns
        if c
        in [
            "order_volume",
            "employee_count",
            "num_items",
            "num_modifiers",
            "hour",
        ]
        or c.startswith("orderType.label_")
        or c.startswith("day_of_week_")
    ]

    if len(df_ml) > 10 and len(features) > 0:
        X = df_ml[features].fillna(0)
        y = df_ml["processing_time_min"]

        rf = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=8
        )
        rf.fit(X, y)

        importance_df = pd.DataFrame(
            {"Feature": features, "Importance": rf.feature_importances_}
        ).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Relative Importance of Each Driver on Processing Time",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.subheader("🔮 Interactive Scenario Simulator ('What-If' Estimator)")
        st.caption(
            "Adjust all 7 variables simultaneously to simulate predicted processing time:"
        )

        sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
        sim_vol = sim_c1.number_input(
            "Hourly Order Volume",
            min_value=1,
            max_value=100,
            value=int(df["order_volume"].median()),
        )
        sim_emp = sim_c2.number_input(
            "Employees Working",
            min_value=1,
            max_value=30,
            value=int(df["employee_count"].median()),
        )
        sim_items = sim_c3.number_input(
            "Items in Order",
            min_value=1,
            max_value=20,
            value=int(df["num_items"].median()),
        )
        sim_mods = sim_c4.number_input(
            "Modifiers in Order",
            min_value=0,
            max_value=20,
            value=int(df["num_modifiers"].median()),
        )

        sim_c5, sim_c6, sim_c7 = st.columns(3)
        sim_hour = sim_c5.slider("Hour of Day", 0, 23, 12)
        sim_day = sim_c6.selectbox("Day of Week", days_order, index=4)
        sim_type = sim_c7.selectbox("Order Type", order_types, index=0)

        # Build single row prediction vector matching feature names
        input_dict = {f: [0] for f in features}
        if "order_volume" in input_dict:
            input_dict["order_volume"] = [sim_vol]
        if "employee_count" in input_dict:
            input_dict["employee_count"] = [sim_emp]
        if "num_items" in input_dict:
            input_dict["num_items"] = [sim_items]
        if "num_modifiers" in input_dict:
            input_dict["num_modifiers"] = [sim_mods]
        if "hour" in input_dict:
            input_dict["hour"] = [sim_hour]

        day_col = f"day_of_week_{sim_day}"
        if day_col in input_dict:
            input_dict[day_col] = [1]

        type_col = f"orderType.label_{sim_type}"
        if type_col in input_dict:
            input_dict[type_col] = [1]

        input_df = pd.DataFrame(input_dict)[features]
        predicted_time = rf.predict(input_df)[0]

        st.metric(
            "Estimated Processing Time",
            f"{predicted_time:.2f} Minutes",
            delta=f"{predicted_time - filtered_df['processing_time_min'].mean():.2f} min vs avg",
            delta_color="inverse",
        )
    else:
        st.info("Not enough data points after filtering to fit the model.")
