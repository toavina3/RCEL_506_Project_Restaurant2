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
# ------------------------------------------------------------------------------
# TAB 1: RESTAURANT FLOW SIMULATION
# ------------------------------------------------------------------------------
with tab1:

    st.subheader("🍽️ Restaurant Order Flow Simulation")

    st.caption(
        "Each dot represents an order moving through a simulated restaurant layout. "
        "The dot speed is inversely related to processing time: faster orders move faster, "
        "while slower orders move more slowly."
    )

    # ==========================================================================
    # 1. SELECT A DATE FOR THE SIMULATION
    # ==========================================================================

    available_dates = sorted(filtered_df["date"].dropna().unique())

    selected_date = st.selectbox(
        "📅 Select a Date to Simulate",
        options=available_dates,
        index=0,
        format_func=lambda x: pd.Timestamp(x).strftime("%A, %B %d, %Y"),
    )

    sim_df = filtered_df[
        filtered_df["date"] == selected_date
    ].copy()

    if sim_df.empty:
        st.warning("No orders are available for the selected date.")
        st.stop()

    # Remove duplicate order IDs if necessary
    sim_df = sim_df.drop_duplicates(subset=["id_x"]).copy()

    # Make sure processing time is valid
    sim_df = sim_df[
        sim_df["processing_time_min"].notna()
        & (sim_df["processing_time_min"] >= 0)
    ].copy()

    if sim_df.empty:
        st.warning("No valid orders are available for this date.")
        st.stop()

    # ==========================================================================
    # 2. RESTAURANT LAYOUT
    # ==========================================================================

    st.markdown("### 🗺️ Restaurant Layout")

    st.caption(
        "The restaurant path is a visualization model rather than an observed "
        "physical route, because the dataset contains order creation and completion "
        "times but does not contain timestamps for individual kitchen stations."
    )

    # Restaurant coordinates
    # The same normalized route is used for every order so that processing time
    # directly determines movement speed.

    route_points = np.array([
        [0.08, 0.50],   # Order received
        [0.25, 0.50],   # Order queue
        [0.43, 0.70],   # Kitchen entrance
        [0.58, 0.70],   # Kitchen / preparation
        [0.72, 0.70],   # Kitchen exit
        [0.82, 0.50],   # Pickup
        [0.93, 0.50],   # Exit
    ])

    # Calculate cumulative distance along route
    segment_lengths = np.sqrt(
        np.sum(np.diff(route_points, axis=0) ** 2, axis=1)
    )

    cumulative_lengths = np.concatenate(
        [[0], np.cumsum(segment_lengths)]
    )

    total_route_length = cumulative_lengths[-1]

    # ==========================================================================
    # 3. HELPER FUNCTION: POSITION ALONG RESTAURANT ROUTE
    # ==========================================================================

    def get_position(progress):

        progress = float(np.clip(progress, 0, 1))

        target_distance = progress * total_route_length

        # Find the segment containing the target distance
        segment_index = np.searchsorted(
            cumulative_lengths,
            target_distance,
            side="right"
        ) - 1

        segment_index = max(
            0,
            min(segment_index, len(route_points) - 2)
        )

        segment_start_distance = cumulative_lengths[segment_index]

        segment_length = segment_lengths[segment_index]

        if segment_length == 0:
            local_progress = 0
        else:
            local_progress = (
                target_distance - segment_start_distance
            ) / segment_length

        start = route_points[segment_index]
        end = route_points[segment_index + 1]

        position = start + local_progress * (end - start)

        return position

    # ==========================================================================
    # 4. PREPARE ORDER TIMESTAMPS
    # ==========================================================================

    sim_df["createdTime"] = pd.to_datetime(
        sim_df["createdTime"]
    )

    sim_df["modifiedTime"] = pd.to_datetime(
        sim_df["modifiedTime"]
    )

    # Remove orders with missing timestamps
    sim_df = sim_df[
        sim_df["createdTime"].notna()
        & sim_df["modifiedTime"].notna()
    ].copy()

    # Ensure completion occurs after creation
    sim_df = sim_df[
        sim_df["modifiedTime"] >= sim_df["createdTime"]
    ].copy()

    if sim_df.empty:
        st.warning("No valid order timelines are available for this date.")
        st.stop()

    # ==========================================================================
    # 5. ORDER TYPE COLORS
    # ==========================================================================

    order_types_sim = sorted(
        sim_df["orderType.label"]
        .fillna("Unknown")
        .astype(str)
        .unique()
    )

    # Plotly-compatible qualitative colors
    color_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    type_colors = {
        order_type: color_palette[i % len(color_palette)]
        for i, order_type in enumerate(order_types_sim)
    }

    sim_df["order_type_display"] = (
        sim_df["orderType.label"]
        .fillna("Unknown")
        .astype(str)
    )

    sim_df["marker_color"] = sim_df[
        "order_type_display"
    ].map(type_colors)

    # ==========================================================================
    # 6. SIMULATION TIME RANGE
    # ==========================================================================

    start_time = sim_df["createdTime"].min().floor("min")
    end_time = sim_df["modifiedTime"].max().ceil("min")

    # Use 1-minute increments for smooth enough movement without generating
    # thousands of frames.
    time_points = pd.date_range(
        start=start_time,
        end=end_time,
        freq="1min"
    )

    # If there are too many frames, increase the interval automatically.
    if len(time_points) > 900:

        time_points = pd.date_range(
            start=start_time,
            end=end_time,
            freq="2min"
        )

    # ==========================================================================
    # 7. CALCULATE EMPLOYEES AND ORDER VOLUME FOR THE SELECTED DATE
    # ==========================================================================

    hourly_context = (
        sim_df.groupby("hour")
        .agg(
            order_volume=("id_x", "nunique"),
            employee_count=("employee.id", "nunique"),
        )
        .reset_index()
    )

    # ==========================================================================
    # 8. INITIAL ORDER POSITIONS
    # ==========================================================================

    initial_time = time_points[0]

    initial_positions = []

    for _, order in sim_df.iterrows():

        created = order["createdTime"]
        completed = order["modifiedTime"]

        if initial_time < created:
            # Order has not entered yet
            x = np.nan
            y = np.nan

        elif completed == created:
            # Instantaneous order
            x, y = route_points[-1]

        elif initial_time >= completed:
            # Already completed
            x, y = route_points[-1]

        else:
            progress = (
                (initial_time - created).total_seconds()
                /
                (completed - created).total_seconds()
            )

            x, y = get_position(progress)

        initial_positions.append((x, y))

    initial_x = [p[0] for p in initial_positions]
    initial_y = [p[1] for p in initial_positions]

    # ==========================================================================
    # 9. HOVER INFORMATION
    # ==========================================================================

    hover_text = []

    for _, order in sim_df.iterrows():

        hover_text.append(
            "<b>Order</b>: " + str(order["id_x"]) +
            "<br><b>Order Type</b>: " + str(order["orderType.label"]) +
            "<br><b>Processing Time</b>: "
            + f"{order['processing_time_min']:.2f} min" +
            "<br><b>Items</b>: " + str(order["num_items"]) +
            "<br><b>Modifiers</b>: " + str(order["num_modifiers"]) +
            "<br><b>Order Volume</b>: " + str(order["order_volume"]) +
            "<br><b>Employees</b>: " + str(order["employee_count"]) +
            "<br><b>Day</b>: " + str(order["day_of_week"]) +
            "<br><b>Hour</b>: " + str(order["hour"]) +
            "<br><b>Created</b>: "
            + order["createdTime"].strftime("%H:%M:%S") +
            "<br><b>Completed</b>: "
            + order["modifiedTime"].strftime("%H:%M:%S") +
            "<extra></extra>"
        )

    # ==========================================================================
    # 10. CREATE RESTAURANT FLOOR PLAN
    # ==========================================================================

    fig = go.Figure()

    # --------------------------------------------------------------------------
    # Restaurant floor areas
    # --------------------------------------------------------------------------

    # Entrance / order received
    fig.add_shape(
        type="rect",
        x0=0.02,
        x1=0.16,
        y0=0.36,
        y1=0.64,
        line=dict(width=2),
        fillcolor="rgba(100, 149, 237, 0.15)",
    )

    # Order queue
    fig.add_shape(
        type="rect",
        x0=0.18,
        x1=0.32,
        y0=0.36,
        y1=0.64,
        line=dict(width=2),
        fillcolor="rgba(255, 193, 7, 0.15)",
    )

    # Kitchen
    fig.add_shape(
        type="rect",
        x0=0.35,
        x1=0.75,
        y0=0.57,
        y1=0.83,
        line=dict(width=2),
        fillcolor="rgba(220, 53, 69, 0.15)",
    )

    # Dining area
    fig.add_shape(
        type="rect",
        x0=0.35,
        x1=0.75,
        y0=0.15,
        y1=0.43,
        line=dict(width=2),
        fillcolor="rgba(40, 167, 69, 0.12)",
    )

    # Pickup area
    fig.add_shape(
        type="rect",
        x0=0.76,
        x1=0.87,
        y0=0.36,
        y1=0.64,
        line=dict(width=2),
        fillcolor="rgba(111, 66, 193, 0.15)",
    )

    # Exit
    fig.add_shape(
        type="rect",
        x0=0.89,
        x1=0.98,
        y0=0.36,
        y1=0.64,
        line=dict(width=2),
        fillcolor="rgba(23, 162, 184, 0.15)",
    )

    # --------------------------------------------------------------------------
    # Area labels
    # --------------------------------------------------------------------------

    fig.add_annotation(
        x=0.09,
        y=0.50,
        text="<b>ORDER<br>RECEIVED</b>",
        showarrow=False,
        font=dict(size=11),
    )

    fig.add_annotation(
        x=0.25,
        y=0.50,
        text="<b>QUEUE</b>",
        showarrow=False,
        font=dict(size=11),
    )

    fig.add_annotation(
        x=0.55,
        y=0.70,
        text="<b>🍳 KITCHEN / PREPARATION</b>",
        showarrow=False,
        font=dict(size=13),
    )

    fig.add_annotation(
        x=0.55,
        y=0.29,
        text="<b>DINING AREA</b>",
        showarrow=False,
        font=dict(size=12),
    )

    fig.add_annotation(
        x=0.815,
        y=0.50,
        text="<b>PICKUP</b>",
        showarrow=False,
        font=dict(size=10),
    )

    fig.add_annotation(
        x=0.935,
        y=0.50,
        text="<b>EXIT</b>",
        showarrow=False,
        font=dict(size=10),
    )

    # --------------------------------------------------------------------------
    # Route line
    # --------------------------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=route_points[:, 0],
            y=route_points[:, 1],
            mode="lines",
            line=dict(
                width=3,
                dash="dot",
            ),
            name="Order Route",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # ==========================================================================
    # 11. ORDER DOTS
    # ==========================================================================

    marker_sizes = (
        10 + sim_df["num_items"].fillna(1).clip(0, 10) * 2
    )

    fig.add_trace(
        go.Scatter(
            x=initial_x,
            y=initial_y,
            mode="markers",
            name="Orders",
            marker=dict(
                size=marker_sizes,
                color=sim_df["marker_color"],
                opacity=0.9,
                line=dict(
                    width=1,
                ),
            ),
            text=hover_text,
            hovertemplate="%{text}",
            customdata=sim_df[
                [
                    "id_x",
                    "processing_time_min",
                    "num_items",
                    "num_modifiers",
                    "order_volume",
                    "employee_count",
                    "orderType.label",
                ]
            ].fillna("N/A").values,
        )
    )

    # ==========================================================================
    # 12. LEGEND FOR ORDER TYPES
    # ==========================================================================

    for order_type in order_types_sim:

        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=12,
                    color=type_colors[order_type],
                ),
                name=str(order_type),
                hoverinfo="skip",
            )
        )

    # ==========================================================================
    # 13. CREATE ANIMATION FRAMES
    # ==========================================================================

    frames = []

    for current_time in time_points:

        x_positions = []
        y_positions = []

        active_order_indices = []

        for i, (_, order) in enumerate(sim_df.iterrows()):

            created = order["createdTime"]
            completed = order["modifiedTime"]

            # Order has not arrived yet
            if current_time < created:
                x_positions.append(np.nan)
                y_positions.append(np.nan)
                continue

            # Order is completed
            if current_time >= completed:
                x_positions.append(np.nan)
                y_positions.append(np.nan)
                continue

            # --------------------------------------------------------------
            # CORE IDEA:
            #
            # progress = elapsed time / processing time
            #
            # Therefore:
            #
            # short processing time -> high speed
            # long processing time  -> low speed
            # --------------------------------------------------------------

            total_seconds = (
                completed - created
            ).total_seconds()

            elapsed_seconds = (
                current_time - created
            ).total_seconds()

            if total_seconds <= 0:
                progress = 1.0
            else:
                progress = elapsed_seconds / total_seconds

            progress = np.clip(progress, 0, 1)

            x, y = get_position(progress)

            x_positions.append(x)
            y_positions.append(y)

            active_order_indices.append(i)

        # Determine current hour
        current_hour = current_time.hour

        current_hour_data = hourly_context[
            hourly_context["hour"] == current_hour
        ]

        if not current_hour_data.empty:

            current_volume = int(
                current_hour_data.iloc[0]["order_volume"]
            )

            current_staff = int(
                current_hour_data.iloc[0]["employee_count"]
            )

        else:

            current_volume = 0
            current_staff = 0

        # Active orders
        active_count = len(active_order_indices)

        frames.append(
            go.Frame(
                name=current_time.strftime("%Y-%m-%d %H:%M"),
                data=[
                    # Route line
                    go.Scatter(
                        x=route_points[:, 0],
                        y=route_points[:, 1],
                    ),

                    # Moving orders
                    go.Scatter(
                        x=x_positions,
                        y=y_positions,
                        mode="markers",
                        marker=dict(
                            size=marker_sizes,
                            color=sim_df["marker_color"],
                            opacity=0.9,
                            line=dict(width=1),
                        ),
                        text=hover_text,
                        hovertemplate="%{text}",
                    ),

                    # Empty traces for the legend
                    *[
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                        )
                        for _ in order_types_sim
                    ],
                ],
                layout=go.Layout(
                    title=dict(
                        text=(
                            f"<b>Restaurant Order Flow</b>"
                            f"<br><sup>"
                            f"{current_time.strftime('%A, %B %d, %Y — %I:%M %p')}"
                            f" | Active Orders: {active_count}"
                            f" | Hourly Volume: {current_volume}"
                            f" | Staff: {current_staff}"
                            f"</sup>"
                        )
                    )
                ),
            )
        )

    fig.frames = frames

    # ==========================================================================
    # 14. PLAY / PAUSE CONTROLS
    # ==========================================================================

    fig.update_layout(

        height=650,

        xaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
        ),

        yaxis=dict(
            range=[0, 1],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),

        plot_bgcolor="white",

        margin=dict(
            l=20,
            r=20,
            t=100,
            b=100,
        ),

        legend=dict(
            title="Order Type",
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),

        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=-0.08,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {
                                    "duration": 80,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                            },
                        ],
                    ),
                ],
            )
        ],

        sliders=[
            dict(
                active=0,
                x=0.18,
                y=-0.08,
                len=0.75,
                xanchor="left",
                yanchor="top",

                currentvalue=dict(
                    prefix="Simulation Time: ",
                    visible=True,
                    xanchor="center",
                ),

                transition=dict(
                    duration=0,
                ),

                steps=[
                    dict(
                        label=t.strftime("%H:%M"),
                        method="animate",
                        args=[
                            [t.strftime("%Y-%m-%d %H:%M")],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    )
                    for t in time_points
                ],
            )
        ],
    )

    # ==========================================================================
    # 15. DISPLAY ANIMATION
    # ==========================================================================

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
        },
    )

    # ==========================================================================
    # 16. EXPLANATION
    # ==========================================================================

    st.markdown("---")

    st.subheader("💡 How to Read This Simulation")

    explanation_col1, explanation_col2, explanation_col3 = st.columns(3)

    with explanation_col1:
        st.markdown(
            """
            **🔵 Each dot = one order**

            - Dot size = number of items
            - Dot color = order type
            - Hover = detailed order information
            """
        )

    with explanation_col2:
        st.markdown(
            """
            **⚡ Speed = processing efficiency**

            An order with a **5-minute processing time**
            travels through the restaurant much faster than
            an order requiring **30 minutes**.
            """
        )

    with explanation_col3:
        st.markdown(
            """
            **🏪 Restaurant workload**

            Watch how many orders are simultaneously
            inside the restaurant. More simultaneous dots
            represent greater operational workload.
            """
        )
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
