import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict

# Set page layout to wide
st.set_page_config(layout="wide")

# Read the CSV file from the public S3 URL
data = pd.read_csv("https://behaviorally-testing.s3.amazonaws.com/behv107_sankey_relevant_session_v1.csv")
data = data[data['image_data'] != 'Yes']
# random commit

st.title("Behaviorally Sankey beh_v107")
st.write("Sankey Diagram of Participant Journeys")

# App selection dropdown
app_names = {
    'com.ss.android.ugc.trill': 'TikTok',
    'com.shopee.ph': 'Shopee',
    'com.lazada.android': 'Lazada'
}
selected_app = st.selectbox("Select App", options=list(app_names.keys()), format_func=lambda x: app_names[x])

# Filter the data after app selection
if selected_app:
    app_data = data[data['apppackagename'] == selected_app]
    print(app_data)

    app_data['participantId'] = app_data['participantId'].astype('str')

    # Order the participant_ids numerically and add an "All" option
    participant_ids = sorted(app_data['participantId'].unique())
    participant_ids = ['All'] + participant_ids  # Add "All" option at the beginning

    # Filter dropdown for participantId
    selected_participant = st.selectbox("Select Participant", participant_ids)

    # Third filter for journey type
    journey_types = ["Purchase", "Non-Purchase"]
    selected_journey_type = st.selectbox("Select Journey Type", journey_types, index=0)  # "Purchase" is default

    # Function to process the journeys until purchases
    def journeys_until_first_purchase(pages):
        if not pages:
            return []
    
        journeys = []
        current_journey = []
    
        for i in range(len(pages)):
            if not current_journey or pages[i] != current_journey[-1]:  # Avoid duplicates
                current_journey.append(pages[i])
    
            if pages[i] == 'Purchase':  # Store journey if 'Purchase' is reached
                journeys.append(current_journey[:])  # Store a copy of the journey
                current_journey = []  # Reset journey for the next sequence
    
        if current_journey:  # Capture any remaining journey if no purchase occurred
            journeys.append(current_journey)
    
        return journeys

    def get_sankey_format_data(df):
        # Exploding the 'url_path' into separate rows
        df_exploded = df.explode('pagetype')

        # Adding Step column based on the order within each session
        df_exploded['Step'] = df_exploded.groupby(['participantId', 'session']).cumcount()

        # Pivoting the DataFrame to get Step columns
        df_pivoted = df_exploded.pivot(index=['participantId', 'session'], columns='Step', values='pagetype')

        # Renaming the columns to match the desired Step format
        df_pivoted.columns = [f'Step{col}' for col in df_pivoted.columns]

        # Resetting index to flatten the DataFrame
        df_pivoted.reset_index(inplace=True)

        # Reordering the columns to match the desired format
        df_final = df_pivoted[['participantId', 'session'] + [f'Step{i}' for i in range(df_pivoted.shape[1] - 2)]]

        return df_final

    def get_first_and_last_five_journeys(list_items):
        if len(list_items) > 10:
            return list_items[:5] + list_items[-5:]
        return list_items

    def get_journeys_until_first_purchase(df, app_package_name, participant_id):
        if participant_id == 'All':
            app_df = df[df['apppackagename'] == app_package_name]
        else:
            app_df = df[(df['apppackagename'] == app_package_name) & (df['participantId'] == participant_id)]
        
        app_df['pagetype'] = app_df['pagetype'].str.split('|')
        app_df = app_df.explode('pagetype',ignore_index=True)
        app_df = app_df[app_df['pagetype'] != 'Viewedrecommendedproduct']
        app_df['pagetype']= app_df['pagetype'].replace('Cart','Cart Journey')
        app_df['eventtime'] = pd.to_datetime(app_df['eventtime'])
        app_df.sort_values(by=['participantId','eventtime','session'], inplace=True)
        aggregated_data = app_df.groupby(['participantId','session']).agg({'pagetype':list}).reset_index()
        aggregated_data['pagetype'] = aggregated_data['pagetype'].apply(journeys_until_first_purchase)
        aggregated_data = aggregated_data.explode('pagetype').reset_index(drop=True)
        aggregated_data['session'] = aggregated_data['session'].astype(str) + '_' + aggregated_data.index.astype('str')
        aggregated_data = aggregated_data[~aggregated_data['pagetype'].isnull()]
        aggregated_data['pagetype'] = aggregated_data['pagetype'].apply(lambda x: x + ['Non-Purchase'] if x[-1] != 'Purchase' else x)
        aggregated_data['pagetype_length'] = aggregated_data['pagetype'].apply(len)
        aggregated_data = aggregated_data[aggregated_data['pagetype_length'] > 1]
        aggregated_data['pagetype'] = aggregated_data['pagetype'].apply(get_first_and_last_five_journeys)
        aggregated_data['journey_type'] = aggregated_data['pagetype'].apply(lambda x: 'Non Purchase Journeys' if x[-1] != 'Purchase' else 'Purchase Journeys')
        aggregated_data['pagetype_length'] = aggregated_data['pagetype'].apply(len)

        # Filter data based on selected journey type
        if selected_journey_type == "Purchase":
            filtered_df = aggregated_data[aggregated_data['journey_type'] == 'Purchase Journeys']
        else:
            filtered_df = aggregated_data[aggregated_data['journey_type'] == 'Non Purchase Journeys']

        return get_sankey_format_data(filtered_df)

    # Filter the data based on the selected participant and journey type
    purchase_paths_df = get_journeys_until_first_purchase(data, selected_app, selected_participant)

    # Set the categories
    purchase_paths_df.loc[purchase_paths_df['participantId'].str.startswith(('1','2','3')),'Category'] = 'IMF'
    purchase_paths_df.loc[purchase_paths_df['participantId'].str.startswith(('4','5','6')),'Category'] = 'PMD'

    # Category selection dropdown
    available_categories = purchase_paths_df['Category'].dropna().unique().tolist()
    selected_category = st.selectbox("Select Category", options=['All'] + available_categories)

    # Filter the DataFrame based on selected category
    if selected_category != 'All':
        purchase_paths_df = purchase_paths_df[purchase_paths_df['Category'] == selected_category]

    event_colors = {
        "Home": "#d02f80",
        "Search": "#d98c26",
        "Review": "#abd629",
        "Category": "#68d22d",
        "Product": "#2bd4bd",
        "Cart Journey": "#229cdd",
        "Checkout":"#229ddd",
        "Purchase": "#964db2",
        "Videolive": "#9a7965",
        "Videononlive": "#9a7345",
        "Voucher": "#6e918b",
        "History": "#edda12",
        "Brandshop": "#64739b",
        "Me":"#63d6d6",
        "Non-Purchase": "#63d8d6",
        "Lazmart":"#23d9d6",
        "Account":"#62d8d6",
        "Shopmain":"#52d8d6",
        "Shopeemall":"#93d8d6",
        "Lazlive":"#12d6d2",
        "Channels":"#5a7965",
        "Voucherhub":"#7bd4bd",
        "IMF": "#ff6f59", 
        "PMD": "#457b9d"
    }
 
    # Initialize lists for sources, targets, values, and colors
    source = []
    target = []
    value = []
    link_colors = []

    # Create dictionaries to store node labels and indices
    node_labels = []
    node_indices = {}
    node_colors = []

    # Step-wise counts for %
    # Count each node occurrence per step
    step_counts = defaultdict(lambda: defaultdict(int))

    for _, row in purchase_paths_df.iterrows():
        steps = [row[col] for col in purchase_paths_df.columns if col.startswith("Step") and pd.notna(row[col])]
        for i, val in enumerate(steps):
            step_counts[i][val] += 1

    # Only sum up non-"Purchase" counts per step for % calculations
    step_totals = {
        i: sum(count for label, count in label_counts.items() if label != "Purchase")
        for i, label_counts in step_counts.items()
    }


    # Node and flow metadata
    source, target, value, link_colors = [], [], [], []
    node_labels, node_indices, node_colors = [], {}, []

    def get_node_index(label):
        if label not in node_indices:
            if "_" in label:
                step_part, page = label.split("_", 1)
                if step_part.startswith("Step") and step_part[4:].isdigit():
                    step_idx = int(step_part[4:])
                    count = step_counts[step_idx][page]
                    total = step_totals[step_idx]
                    pct = ""
                    if label != "Purchase" and total > 0:
                        pct = f" ({round((count / total) * 100):.0f}%)"
                    display_label = f"{page}{pct}"
                else:
                    display_label = label
            else:
                display_label = label
            node_indices[label] = len(node_labels)
            node_labels.append(display_label)
            key = label.split("_")[1] if "_" in label else label
            node_colors.append(event_colors.get(key, "grey"))
        return node_indices[label]


    # Build source-target-value lists
    for _, row in purchase_paths_df.iterrows():
        steps = [row[col] for col in purchase_paths_df.columns if col.startswith("Step") and pd.notna(row[col])]
        for i in range(len(steps) - 1):
            src = get_node_index(f"Step{i}_{steps[i]}")
            tgt = get_node_index(f"Step{i+1}_{steps[i+1]}" if steps[i+1] not in ['Purchase', 'Non-Purchase'] else steps[i+1])
            source.append(src)
            target.append(tgt)
            value.append(1)
            link_colors.append("rgba(200,200,200,0.3)")

    # # Build a dataframe to show all node counts and percentages per step
    # rows = []
    # for step_idx, label_counts in step_counts.items():
    #     total = sum(label_counts.values())
    #     for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
    #         percent = round(count / total * 100) if total else 0
    #         rows.append({
    #             "Step": f"Step {step_idx}",
    #             "Node": label,
    #             "Count": count,
    #             "Percent": f"{percent}%",
    #         })
    #     rows.append({
    #         "Step": f"Step {step_idx}",
    #         "Node": "TOTAL",
    #         "Count": total,
    #         "Percent": "100%"
    #     })

    # step_df = pd.DataFrame(rows)
            
    # Sankey figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="rgba(0,0,0,0)", width=0),
            label=node_labels,
            color=node_colors
        ),
        link=dict(source=source, target=target, value=value, color=link_colors)
    ))

    fig.update_layout(
        title_text="E-commerce Purchase Journeys (First 5 steps, Last 5 steps only)",
        template="none",
        font_family="Helvetica",
        font_size=20,
        font_color="black",
        margin=dict(l=50, r=50, t=100, b=100),
        width=1600,
        height=1200
    )

    st.plotly_chart(fig, use_container_width=True)