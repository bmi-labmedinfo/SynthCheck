import base64
import io
import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import scipy.stats as stats
import umap
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from pyhtml2pdf import converter
from scipy.spatial import distance
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

global real_data, synthetic_data, dict_features_type, num_features, cat_features, real_train_data, real_test_data, encoder_labels

real_data = pd.DataFrame()

synthetic_data = pd.DataFrame()

dict_features_type = {}

real_train_data = pd.DataFrame()
real_test_data = pd.DataFrame()

global path_user
path_user = os.getcwd().replace('\\', '/')
if not os.path.exists(os.path.join(path_user, 'data_figures')):
    os.makedirs(os.path.join(path_user, 'data_figures'))
if not os.path.exists(os.path.join(path_user, 'data_report')):
    os.makedirs(os.path.join(path_user, 'data_report'))

print("!! START !!")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,
                prevent_initial_callbacks="initial_duplicate")

# PAGE LAYAOUT (static elements)
app.layout = html.Div([

    # Header application (top)
    dbc.NavbarSimple([
        html.H1("Evaluation Metrics", style={"color": "#ffffff"})
    ],
        sticky="top",
        color="primary",
        dark=True,
        style={"height": "10vh"}
    ),

    # Actual user position
    dcc.Location(id='user_position', refresh=False),

    # Page content
    html.Div(id='page_content', style={'minHeight': '82vh'}),

    # Navigation bar (bottom)
    dbc.NavbarSimple([
        dbc.NavItem(dbc.NavLink('Load Data', href='/page_1'), style={"marginLeft": "2vw", "marginRight": "5vw"}),

        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem('Univariate Resemblance Analysis', href='/page_2_ura'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Multivariate Relationships Analysis', href='/page_2_mra'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Data Labeling Analysis', href='/page_2_dla'),
            ],
            style={"marginRight": "5vw"},
            menu_variant="dark",
            direction='up',
            nav=True,
            in_navbar=True,
            disabled=True,
            id="nav2",
            label="Resemblance"),

        dbc.NavItem(dbc.NavLink('Utility', href='/page_3', id="nav3", disabled=True), style={"marginRight": "5vw"}),

        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem('Similarity Evaluation Analysis', href='/page_4_sea'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Membership Inference Attack', href='/page_4_mia'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Attribute Inference Attack', href='/page_4_aia'),
            ],
            style={"marginRight": "3vw"},
            menu_variant="dark",
            direction='up',
            nav=True,
            in_navbar=True,
            disabled=True,
            id="nav4",
            label="Privacy"),

        dbc.NavItem(dbc.Button(["Download Report"], id="button-report", color="primary", disabled=True),
                    id="item-report"),
        dcc.Loading([dcc.Download(id="download-report")], type="circle", fullscreen=True),

    ],
        style={"height": "8vh"},
        color="primary",
        dark=True,
        sticky="bottom",
        links_left=True
    )

])


@app.callback(Output('item-report', 'style'),
              Input('user_position', 'pathname'))
def update_download_visibility(pathname):
    if pathname == "/page_1" or pathname == "/":
        return {'display': 'none'}
    else:
        return {"display": "block", "position": "relative", "left": "30vw"}


@app.callback(Output('page_content', 'children'),
              Input('user_position', 'pathname'))
def display_page(pathname):
    if pathname == '/page_1':
        return page_1
    elif pathname == '/page_2_ura':
        return page_2_ura
    elif pathname == '/page_2_mra':
        return page_2_mra
    elif pathname == '/page_2_dla':
        return page_2_dla
    elif pathname == '/page_3':
        return page_3
    elif pathname == '/page_4_sea':
        return page_4_sea
    elif pathname == '/page_4_mia':
        return page_4_mia
    elif pathname == '/page_4_aia':
        return page_4_aia
    else:
        return page_1


# PAGE 1 CONTENTS (load data)
page_1 = html.Div([
    dbc.Container([

        # Upload real data
        dbc.Row([dbc.Col([
            html.H2("Real dataset", style={'margin-left': '1vw', 'margin-top': '1vw'}),
            dcc.Upload(
                id='upload-data-real',
                children=html.Div([
                    'Drag and drop the file here or ',
                    html.A('select a file')
                ]),
                style={
                    'width': '80vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '5vw',
                    'margin-right': '5vw'
                },
                multiple=False
            ),
            dcc.Loading(html.Div(id='output-data-upload-real', style={'margin-bottom': '1.5vw'})),
        ])]),

        dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)]),

        # Upload synthetic data
        dbc.Row([dbc.Col([
            html.H2("Synthetic dataset", style={'margin-left': '1vw'}),
            dcc.Upload(
                id='upload-data-syn',
                children=html.Div([
                    'Drag and drop the file here or ',
                    html.A('select a file')
                ]),
                style={
                    'width': '80vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '5vw',
                    'margin-right': '5vw'
                },
                multiple=False
            ),
            dcc.Loading(html.Div(id='output-data-upload-syn', style={'margin-bottom': '1.5vw'})),
        ])]),

        dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)]),

        # Upload features type
        dbc.Row([dbc.Col([
            html.H2("Features data types", style={'margin-left': '1vw'}),
            dcc.Upload(
                id='upload-data-type',
                children=html.Div([
                    'Drag and drop the file here or ',
                    html.A('select a file')
                ]),
                style={
                    'width': '80vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '5vw',
                    'margin-right': '5vw'
                },
                multiple=False
            ),
            html.Div(id='output-data-upload-type', style={'margin-bottom': '1.5vw'})
        ])]),

        dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)]),

    ], fluid=True)
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    df = pd.read_csv(io.StringIO(decoded))

    return df


@app.callback(Output('output-data-upload-real', 'children'),
              Input('upload-data-real', 'contents'),
              State('upload-data-real', 'filename'))
def update_table_real(contents, filename):
    global real_data, synthetic_data
    if contents is not None:
        real_data = parse_contents(contents, filename)

        global encoder_labels
        encoder_labels = LabelEncoder()

        for col in real_data.columns:
            if real_data[col].dtype == 'object':
                real_data[col] = encoder_labels.fit_transform(real_data[col])

        if not synthetic_data.empty:
            if real_data.shape[0] < synthetic_data.shape[0]:
                synthetic_data = synthetic_data.sample(n=len(real_data), random_state=80).reset_index(drop=True)
            elif real_data.shape[0] > synthetic_data.shape[0]:
                real_data = real_data.sample(n=len(synthetic_data), random_state=80).reset_index(drop=True)

        children = [
            html.Div([
                dash_table.DataTable(
                    data=real_data.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in real_data.columns],
                    page_action='none',
                    style_table={'height': '300px',
                                 'width': '80vw',
                                 'margin-left': '5vw',
                                 'margin-right': '5vw',
                                 'overflowY': 'auto',
                                 'overflowX': 'auto'
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100},
                )
            ])
        ]
        return children


@app.callback(Output('output-data-upload-syn', 'children'),
              Input('upload-data-syn', 'contents'),
              State('upload-data-syn', 'filename'))
def update_table_syn(contents, filename):
    global synthetic_data, real_data
    if contents is not None:
        synthetic_data = parse_contents(contents, filename)

        encoder_label = LabelEncoder()

        for col in real_data.columns:
            if synthetic_data[col].dtype == 'object':
                synthetic_data[col] = encoder_label.fit_transform(synthetic_data[col])

        if not real_data.empty:
            if real_data.shape[0] < synthetic_data.shape[0]:
                synthetic_data = synthetic_data.sample(n=len(real_data), random_state=80).reset_index(drop=True)
            elif real_data.shape[0] > synthetic_data.shape[0]:
                real_data = real_data.sample(n=len(synthetic_data), random_state=80).reset_index(drop=True)

        children = [
            html.Div([
                dash_table.DataTable(
                    data=synthetic_data.to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in synthetic_data.columns],
                    page_action='none',
                    style_table={'height': '300px',
                                 'width': '80vw',
                                 'margin-left': '5vw',
                                 'margin-right': '5vw',
                                 'overflowY': 'auto',
                                 'overflowX': 'auto'
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100},
                )
            ])
        ]
        return children


@app.callback(Output('output-data-upload-type', 'children'),
              Input('upload-data-type', 'contents'),
              State('upload-data-type', 'filename'))
def update_table_type(contents, filename):
    global dict_features_type, num_features, cat_features
    if contents is not None:
        df = parse_contents(contents, filename)
        dict_features_type = df.set_index("Feature")["Type"].to_dict()
        num_features = [key for key, value in dict_features_type.items() if value == "numerical"]
        cat_features = [key for key, value in dict_features_type.items() if value == "categorical"]

        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': "Feature", 'id': "Feature"}, {'name': "Type", 'id': "Type"}],
                    page_action='none',
                    style_table={'height': '300px',
                                 'width': '60vw',
                                 'margin-left': '15vw',
                                 'margin-right': '15vw',
                                 'overflowY': 'auto',
                                 'overflowX': 'auto'
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    # style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100},
                )
            ])
        ]
        return children


@app.callback(Output('nav2', 'disabled'),
              Output('nav3', 'disabled'),
              Output('nav4', 'disabled'),
              Input('output-data-upload-real', 'children'),
              Input('output-data-upload-syn', 'children'),
              Input('output-data-upload-type', 'children'))
def active_nav(r, s, t):
    if not real_data.empty and not synthetic_data.empty and dict_features_type:
        return False, False, False
    else:
        return True, True, True


# PAGE 2 CONTENTS (resemblance metrics)

# URA metrics
page_2_ura = html.Div([

    dbc.Container(
        [
            # header URA section
            dbc.Row(
                dbc.Col(html.H3("Univariate Resemblance Analysis",
                                style={'margin-left': '1vw', 'margin-top': '1vw', 'margin-bottom': '2vw'}),
                        width="auto")),

            # Dropdown URA numerical test
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose statistical test", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-test-num",
                                options=[
                                    {"label": "Kolmogorov–Smirnov test", "value": "ks_test"},
                                    {"label": "Student T-test", "value": "t_test"},
                                    {"label": "Mann Whitney U-test", "value": "u_test"},
                                ],
                                value="ks_test",
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I("numerical")]), width={'size': 'auto'}, align="center")
                ]
            ),

            # Table and graphs URA numerical test
            dbc.Row(
                [
                    dbc.Col(html.Div(id="output-table-pvalue-num"), width={'size': 3, 'offset': 1}, align="center"),
                    dbc.Col(html.Div(id="output-graph-num"), width={'size': 5, 'offset': 2}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 0}),

            # Dropdown URA categorical test
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose statistical test", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-test-cat",
                                options=[
                                    {"label": "Chi-square test", "value": "chi_test"},
                                ],
                                value="chi_test",
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I("categorical")]), width={'size': 'auto'},
                            align="center")
                ]
            ),

            # Table and graphs URA categorical test
            dbc.Row(
                [
                    dbc.Col(html.Div(id="output-table-pvalue-cat"), width={'size': 3, 'offset': 1}, align="center"),
                    dbc.Col(html.Div(id="output-graph-cat"), width={'size': 5, 'offset': 2}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 1}),

            # Dropdown URA metrics distance
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose distance metric", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-dist",
                                options=[
                                    {"label": "Cosine distance", "value": "cos_dist"},
                                    {"label": "Jensen-Shannon distance", "value": "js_dist"},
                                    {"label": "Wasserstein distance", "value": "w_dist"},
                                ],
                                value="cos_dist",
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I("numerical")]), width={'size': 'auto'},
                            align="center")
                ]
            ),

            # Table URA distances
            dbc.Row(
                [
                    dbc.Col(html.Div(id="output-table-dist"), width={'size': 3, 'offset': 1}, align="center"),
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 2}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


def ks_tests(real, synthetic):
    attribute_names = real.columns
    p_values = list()

    for c in attribute_names:
        _, p = stats.ks_2samp(real[c], synthetic[c])
        p_values.append(np.round(p, 5))

    dict_pvalues = dict(zip(attribute_names, p_values))

    return dict_pvalues


def student_t_tests(real, synthetic):
    attribute_names = real.columns
    p_values = list()

    for c in attribute_names:
        _, p = stats.ttest_ind(real[c], synthetic[c])
        p_values.append(np.round(p, 5))

    dict_pvalues = dict(zip(attribute_names, p_values))

    return dict_pvalues


def mann_whitney_tests(real, synthetic):
    attribute_names = real.columns
    p_values = list()

    for c in attribute_names:
        _, p = stats.mannwhitneyu(real[c], synthetic[c])
        p_values.append(np.round(p, 5))

    dict_pvalues = dict(zip(attribute_names, p_values))

    return dict_pvalues


def chi_squared_tests(real, synthetic):
    attribute_names = real.columns
    p_values = list()

    for c in attribute_names:
        observed = pd.crosstab(real[c], synthetic[c])
        _, p, _, _ = stats.chi2_contingency(observed)
        p_values.append(np.round(p, 5))

    dict_pvalues = dict(zip(attribute_names, p_values))

    return dict_pvalues


def cosine_distances(real, synthetic):
    attribute_names = real.columns
    distances = list()

    for c in attribute_names:
        distances.append(distance.cosine(real[c].values, synthetic[c].values))

    dict_distances = dict(zip(attribute_names, np.round(distances, 5)))

    return dict_distances


def js_distances(real, synthetic):
    attribute_names = real.columns
    distances = list()

    for c in attribute_names:
        prob_distribution_real = stats.gaussian_kde(real[c].values).pdf(real[c].values)
        prob_distribution_synthetic = stats.gaussian_kde(real[c].values).pdf(synthetic[c].values)
        distances.append(distance.jensenshannon(prob_distribution_real, prob_distribution_synthetic))

    dict_distances = dict(zip(attribute_names, np.round(distances, 5)))

    return dict_distances


def wass_distances(real, synthetic):
    attribute_names = real.columns
    distances = list()

    for c in attribute_names:
        distances.append(stats.wasserstein_distance(real[c].values, synthetic[c].values))

    dict_distances = dict(zip(attribute_names, np.round(distances, 5)))

    return dict_distances


def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    return pd.DataFrame(scaled, columns=df.columns.tolist())


@app.callback(Output('output-table-pvalue-num', 'children'),
              Output({"type": "data-report", "index": 0}, 'data'),
              Input('dropdown-test-num', 'value'))
def update_table_pvalue_num(value):
    if value is not None:

        if value == 'ks_test':
            dict_pvalues = ks_tests(real_data[num_features], synthetic_data[num_features])
        elif value == 't_test':
            dict_pvalues = student_t_tests(real_data[num_features], synthetic_data[num_features])
        elif value == 'u_test':
            dict_pvalues = mann_whitney_tests(real_data[num_features], synthetic_data[num_features])

        df = pd.DataFrame(list(dict_pvalues.items()), columns=['Feature', 'p value'])

        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': "Feature", 'id': "Feature"}, {'name': "p value", 'id': "p value"}],
                    page_action='none',
                    style_table={'height': '300px',
                                 'overflowY': 'auto',
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{p value} <= 0.05',
                            },
                            'backgroundColor': '#e74c3c',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{p value} > 0.05',
                            },
                            'backgroundColor': '#18bc9c',
                            'color': 'white'
                        },
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': '#003153',
                            'border': '1px solid blue'
                        }
                    ],
                    id='tbl-pvalue-num'
                )
            ])
        ]
        return children, [dict_pvalues, {'selected_opt': value}]


@app.callback(Output('output-table-pvalue-cat', 'children'),
              Output({"type": "data-report", "index": 1}, 'data'),
              Input('dropdown-test-cat', 'value'))
def update_table_pvalue_cat(value):
    if value is not None:

        if value == 'chi_test':
            dict_pvalues = chi_squared_tests(real_data[cat_features], synthetic_data[cat_features])

        df = pd.DataFrame(list(dict_pvalues.items()), columns=['Feature', 'p value'])

        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': "Feature", 'id': "Feature"}, {'name': "p value", 'id': "p value"}],
                    page_action='none',
                    style_table={'height': '300px',
                                 'overflowY': 'auto',
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                    style_data_conditional=[
                        {
                            'if': {
                                'filter_query': '{p value} > 0.05',
                            },
                            'backgroundColor': '#e74c3c',
                            'color': 'white'
                        },
                        {
                            'if': {
                                'filter_query': '{p value} <= 0.05',
                            },
                            'backgroundColor': '#18bc9c',
                            'color': 'white'
                        },
                        {
                            'if': {'state': 'active'},
                            'backgroundColor': '#003153',
                            'border': '1px solid blue'
                        }
                    ],
                    id='tbl-pvalue-cat'
                )
            ])
        ]
        return children, [dict_pvalues, {'selected_opt': value}]


@app.callback(Output('output-table-dist', 'children'),
              Output({"type": "data-report", "index": 2}, 'data'),
              Input('dropdown-dist', 'value'))
def update_table_dist(value):
    if value is not None:

        if value == 'cos_dist':
            dict_distances = cosine_distances(scale_data(real_data[num_features]),
                                              scale_data(synthetic_data[num_features]))
        elif value == 'js_dist':
            dict_distances = js_distances(scale_data(real_data[num_features]),
                                          scale_data(synthetic_data[num_features]))
        elif value == 'w_dist':
            dict_distances = wass_distances(scale_data(real_data[num_features]),
                                            scale_data(synthetic_data[num_features]))

        df = pd.DataFrame(list(dict_distances.items()), columns=['Feature', 'Distance value'])

        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': "Feature", 'id': "Feature"}, {'name': "Distance value", 'id': "Distance value"}],
                    page_action='none',
                    sort_action="native",
                    sort_mode="multi",
                    style_table={'height': '300px',
                                 'overflowY': 'auto',
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                    id='tbl-dist'
                )
            ])
        ]
        return children, [dict_distances, {'selected_opt': value}]


def create_fig_ura(col_real, col_syn, feature_name, feature_type):
    if feature_type == "numerical":
        fig_real = ff.create_distplot([col_real], ['Original Data'], show_hist=False, show_rug=False, colors=['blue'])
        fig_syn = ff.create_distplot([col_syn], ['Synthetic Data'], show_hist=False, show_rug=False, colors=['red'])
        fig_real.update_traces(fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.5)', selector=dict(type='scatter'))
        fig_syn.update_traces(fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.5)', selector=dict(type='scatter'))
        fig = fig_real.add_traces(fig_syn.data)

    else:
        plot_data = pd.DataFrame({
            'value': col_real + col_syn,
            'variable': ['Original Data'] * len(col_real) + ['Synthetic Data'] * len(col_syn)
        })
        fig = px.histogram(plot_data, x="value", color='variable', barmode='group', histnorm='percent',
                           color_discrete_map={'Original Data': 'blue', 'Synthetic Data': 'red'}, opacity=0.9)

    fig.update_layout(title_text=feature_name, showlegend=True)

    return fig


@app.callback(Output('output-graph-num', 'children'),
              Input('tbl-pvalue-num', 'active_cell'))
def update_graphs(active_cell):
    if active_cell is not None:

        selected_feature = num_features[active_cell['row']]

        col_real = real_data[selected_feature].tolist()
        col_syn = synthetic_data[selected_feature].tolist()

        fig = create_fig_ura(col_real, col_syn, selected_feature, "numerical")

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

    else:
        children = [
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [],
                        'layout': {
                            'annotations': [{
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'text': 'Click on an adjacent table cell to view the corresponding chart,<br>which '
                                        'compares real data with synthetic data for the selected feature.',
                                'showarrow': False,
                                'font': {'size': 16},
                                'xanchor': 'center',
                                'yanchor': 'middle',
                                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                                'bordercolor': 'rgba(0, 0, 0, 0.2)',
                                'borderwidth': 1,
                                'borderpad': 4,
                                'borderline': {'width': 2}
                            }],
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'template': 'plotly_white'
                        }
                    },
                    config={'displayModeBar': False}
                )
            ])
        ]

    return children


@app.callback(Output('output-graph-cat', 'children'),
              Input('tbl-pvalue-cat', 'active_cell'))
def update_graphs(active_cell):
    if active_cell is not None:

        selected_feature = cat_features[active_cell['row']]

        col_real = real_data[selected_feature].tolist()
        col_syn = synthetic_data[selected_feature].tolist()

        fig = create_fig_ura(col_real, col_syn, selected_feature, "categorical")

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

    else:
        children = [
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [],
                        'layout': {
                            'annotations': [{
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'text': 'Click on an adjacent table cell to view the corresponding chart,<br>which '
                                        'compares real data with synthetic data for the selected feature.',
                                'showarrow': False,
                                'font': {'size': 16},
                                'xanchor': 'center',
                                'yanchor': 'middle',
                                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                                'bordercolor': 'rgba(0, 0, 0, 0.2)',
                                'borderwidth': 1,
                                'borderpad': 4,
                                'borderline': {'width': 2}
                            }],
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'template': 'plotly_white'
                        }
                    },
                    config={'displayModeBar': False}
                )
            ])
        ]

    return children


# MRA metrics
page_2_mra = html.Div([

    dbc.Container(
        [
            # Header MRA section
            dbc.Row(
                dbc.Col(html.H3("Multivariate Relationship Analysis",
                                style={'margin-left': '1vw', 'margin-top': '1vw', 'margin-bottom': '1vw'}),
                        width="auto")),

            dbc.Row(
                dbc.Col(html.H4("Comparison correlation matrices",
                                style={'margin-left': '1.5vw', 'margin-top': '2vw', 'margin-bottom': '1.5vw'}),
                        width="auto")),

            # Dropdown numerical matrices/categorical matrices, Radio for graphical visualization
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose correlation matrix type", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-corr-mat",
                                options=[
                                    {"label": "Pairwise Pearson correlation matrices", "value": "corr_num"},
                                    {"label": "Normalized contingency tables", "value": "corr_cat"},
                                ],
                                value="corr_num",
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I(id="label-type")]),
                            width={'size': 2}, align="center"),

                    dbc.Col(html.Div(
                        [
                            dbc.RadioItems(
                                id="radios-mat",
                                className="btn-group",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                                options=[
                                    {"label": "Real vs Syn", "value": "rs"},
                                    {"label": "Differences", "value": "diff"},
                                ],
                                value="rs",
                            )
                        ], className="btn-group"
                    ), width={'size': 3}, align="center")
                ]
            ),

            # Correlation matrices
            dbc.Row(
                [
                    dbc.Col(html.Div(id="output-corr-mat"), width={'size': 8, 'offset': 2}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 3}),

            dbc.Row(
                dbc.Col([html.H4(["Comparison Outliers ",
                                  html.Span("ℹ", id="info-outlier", style={'cursor': 'pointer'})],
                                 style={'margin-left': '1.5vw', 'margin-top': '7vw', 'margin-bottom': '0vw'},
                                 id="title-outlier"),
                         dbc.Tooltip(
                             "Unsupervised Outlier Detection using the Local Outlier Factor. It measures the "
                             "local deviation of the density of a given sample with respect to its neighbors.",
                             target="info-outlier",
                         )],
                        width="auto")),

            # Boxplot LOF scores
            dbc.Row(
                [
                    dbc.Col(html.Div(id="output-boxplot"), width={'size': 8, 'offset': 2}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 4}),

            dbc.Row(
                dbc.Col(html.H4("Comparison Principal Component Analysis",
                                style={'margin-left': '1.5vw', 'margin-top': '7vw', 'margin-bottom': '0vw'},
                                id="title-pca"),
                        width="auto")),

            # PCA
            dbc.Row(
                [
                    dbc.Col(html.Div(id="output-pca"), width={'size': 6, 'offset': 1}, align="center"),
                    dbc.Col(html.Div(id="output-table-pca"), width={'size': 3, 'offset': 1}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 5}),

            dbc.Row(
                dbc.Col(html.H4("Comparison UMAP visualization",
                                style={'margin-left': '1.5vw', 'margin-top': '2vw', 'margin-bottom': '1.5vw'}),
                        width="auto")),

            # Radio for UMAP visualization, input meta-parameters
            dbc.Form(
                dbc.Row(
                    [
                        dbc.Col(html.Div(
                            [
                                dbc.RadioItems(
                                    id="radios-umap",
                                    className="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                    labelCheckedClassName="active",
                                    options=[
                                        {"label": "Real vs Syn", "value": "rs"},
                                        {"label": "Together", "value": "tog"},
                                    ],
                                    value="rs",
                                )
                            ], className="radio-group"
                        ), width={'size': 'auto', 'offset': 1}, align="end"),

                        dbc.Col(html.Div(
                            [
                                html.P("Type the number of neighboring"),
                                dbc.Input(type="number", min=2, step=1,
                                          value=20, required="required", id="num-neighbors-input"),
                            ]
                        ), width={'size': 2, 'offset': 1}),

                        dbc.Col(html.Div(
                            [
                                html.P("Type min_dist parameter value"),
                                dbc.Input(type="number", min=0, max=1, step=0.05,
                                          value=0.1, required="required", id="min-dist-input"),
                            ]
                        ), width={'size': 2}),

                        dbc.Col(html.Div(
                            [
                                dbc.Button("Run UMAP", color="info", id="run-umap"),
                            ]
                        ), width={'size': 'auto'}, align="end"),
                    ]
                )
            ),

            dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),

            # Graphs UMAP
            dcc.Loading(dbc.Row(
                [
                    dbc.Col(html.Div(id="output-umap"), width={'size': 8, 'offset': 2}, align="center"),
                ],
            ), id={"type": "load-res", "index": 1}),
            dcc.Store(id={"type": "data-report", "index": 6}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('label-type', 'children'),
              Input('dropdown-corr-mat', 'value'))
def change_label_type(value):
    if value is not None:
        if value == "corr_num":
            return ["numerical"]
        elif value == "corr_cat":
            return ["categorical"]


def check_numerical_correlations(real, synthetic):
    ppc_matrix_real = real.corr(method='pearson')
    ppc_matrix_synthetic = synthetic.corr(method='pearson')
    diff = abs(ppc_matrix_real - ppc_matrix_synthetic)

    return ppc_matrix_real, ppc_matrix_synthetic, diff


def check_categorical_correlations(real, synthetic):
    def get_normalized_conting_table(df):
        factors_paired = [(i, j) for i in df.columns.values for j in df.columns.values]

        chi2 = []

        for f in factors_paired:
            if f[0] != f[1]:
                chitest = chi2_contingency(pd.crosstab(df[f[0]], df[f[1]]))
                chi2.append(chitest[0])
            else:
                chi2.append(0)

        chi2 = np.array(chi2).reshape((df.shape[1], df.shape[1]))
        chi2 = pd.DataFrame(chi2, index=df.columns.values, columns=df.columns.values)

        return (chi2 - np.min(chi2, axis=None)) / np.ptp(chi2, axis=None)

    normalized_chi2_real = get_normalized_conting_table(real)
    normalized_chi2_synthetic = get_normalized_conting_table(synthetic)
    diff = abs(normalized_chi2_real - normalized_chi2_synthetic)

    return normalized_chi2_real, normalized_chi2_synthetic, diff


@app.callback(Output('output-corr-mat', 'children'),
              Output({"type": "data-report", "index": 3}, 'data'),
              Input('dropdown-corr-mat', 'value'),
              Input('radios-mat', 'value'))
def update_graphs(value, radio):
    if value is not None:

        if value == "corr_num":
            mat_real, mat_syn, mat_diff = check_numerical_correlations(real_data[num_features],
                                                                       synthetic_data[num_features])
        elif value == "corr_cat":
            mat_real, mat_syn, mat_diff = check_categorical_correlations(real_data[cat_features],
                                                                         synthetic_data[cat_features])

        if radio == "rs":
            fig1 = px.imshow(mat_real, aspect="auto")
            fig1.update_layout(title_text='Real Data')
            fig2 = px.imshow(mat_syn, aspect="auto")
            fig2.update_layout(title_text='Synthetic Data')

            children = [
                html.Div([
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig1), width=6),
                        dbc.Col(dcc.Graph(figure=fig2), width=6)
                    ])
                ])
            ]
            fig = [fig1, fig2]
        else:
            fig = px.imshow(mat_diff)

            children = [
                html.Div([
                    dcc.Graph(figure=fig)
                ])
            ]

    else:
        children = [
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [],
                        'layout': {
                            'annotations': [{
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'text': 'Choose a correlation matrix type to view the corresponding chart,<br>which '
                                        'compares real data with synthetic data.',
                                'showarrow': False,
                                'font': {'size': 16},
                                'xanchor': 'center',
                                'yanchor': 'middle',
                                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                                'bordercolor': 'rgba(0, 0, 0, 0.2)',
                                'borderwidth': 1,
                                'borderpad': 4,
                                'borderline': {'width': 2}
                            }],
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'template': 'plotly_white'
                        }
                    },
                    config={'displayModeBar': False}
                )
            ])
        ]

    return children, [fig, value, radio]


def check_lof(dataset):
    clf = LocalOutlierFactor(n_neighbors=20)

    labels_out = clf.fit_predict(dataset)

    neg_lof_score = clf.negative_outlier_factor_

    return neg_lof_score, labels_out


@app.callback(Output('output-boxplot', 'children'),
              Output({"type": "data-report", "index": 4}, 'data'),
              Input('title-outlier', 'children'))
def update_graphs(children_in):
    if children_in is not None:
        neg_lof_score_real, _ = check_lof(real_data)
        neg_lof_score_syn, _ = check_lof(synthetic_data)

        fig = go.Figure()
        fig.add_trace(go.Box(x=neg_lof_score_syn, name="Synthetic", marker_color="red"))
        fig.add_trace(go.Box(x=neg_lof_score_real, name="Real", marker_color="blue"))

        fig.update_traces(boxpoints='all', jitter=0.3, pointpos=-1.8, marker_size=6)
        fig.update_layout(xaxis_title="negative LOF score")

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

        return children, fig


@app.callback(Output({"type": "data-report", "index": 5}, 'data'),
              Output('output-pca', 'children'),
              Input('title-pca', 'children'))
def update_graphs(children_in):
    if children_in is not None:
        pca = PCA()
        pca.fit(real_data)
        real_var_ratio_cum = np.cumsum(pca.explained_variance_ratio_ * 100)
        pca = PCA()
        pca.fit(synthetic_data)
        syn_var_ratio_cum = np.cumsum(pca.explained_variance_ratio_ * 100)
        components = list(range(1, len(real_var_ratio_cum) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=components, y=real_var_ratio_cum, mode='lines+markers', name="Real"))
        fig.add_trace(go.Scatter(x=components, y=syn_var_ratio_cum, mode='lines+markers', name="Synthetic"))

        fig.update_layout(xaxis_title="Components", yaxis_title="Explained variance ratio (%)")

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

        return [{'x': components, 'yr': real_var_ratio_cum, 'ys': syn_var_ratio_cum}, fig], children


@app.callback(Output('output-table-pca', 'children'),
              Input({"type": "data-report", "index": 5}, 'data'))
def update_table_pca(data):
    if data is not None:
        data = data[0]

        diff = np.round(abs(np.array(data['yr']) - np.array(data['ys'])), 2)
        df = pd.DataFrame(list(zip(data['x'], diff)), columns=['Component', 'Difference (%)'])

        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': "Component", 'id': "Component"},
                             {'name': "Difference (%)", 'id': "Difference (%)"}],
                    page_action='none',
                    style_table={'height': '40vh',
                                 'overflowY': 'auto',
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                )
            ])
        ]
        return children


@app.callback(Output('output-umap', 'children'),
              Output({"type": "data-report", "index": 6}, 'data'),
              [Input('run-umap', 'n_clicks')],
              [State('num-neighbors-input', 'value'),
               State('min-dist-input', 'value'),
               State('radios-umap', 'value')])
def run_code_on_click(n_clicks, num_neighbors, min_dist, radio):
    if radio == "rs":
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Real Data", "Synthetic Data"])

        reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist, n_components=2)
        embedding_real = reducer.fit_transform(real_data)
        fig.add_trace(go.Scatter(x=embedding_real[:, 0], y=embedding_real[:, 1],
                                 mode='markers', name="Real"), 1, 1)

        reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist, n_components=2)
        embedding_synthetic = reducer.fit_transform(synthetic_data)
        fig.add_trace(go.Scatter(x=embedding_synthetic[:, 0], y=embedding_synthetic[:, 1],
                                 mode='markers', name="Synthetic", marker_color="red"), 1, 2)

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]
    else:
        reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist, n_components=2)
        df = pd.concat([real_data, synthetic_data], ignore_index=True)
        embedding = reducer.fit_transform(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=embedding[0:real_data.shape[0], 0], y=embedding[0:real_data.shape[0], 1],
                                 mode='markers', name="Real"))
        fig.add_trace(go.Scatter(x=embedding[real_data.shape[0]:, 0], y=embedding[real_data.shape[0]:, 1],
                                 mode='markers', name="Synthetic"))

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

    return children, [fig, num_neighbors, min_dist, radio]


# DLA metrics
page_2_dla = html.Div([

    dbc.Container(
        [
            # Header DLA section
            dbc.Row(
                dbc.Col(html.H3("Data Labeling Analysis",
                                style={'margin-left': '1vw', 'margin-top': '1vw', 'margin-bottom': '1vw'}),
                        width="auto")),

            dcc.Loading(dbc.Row([
                dcc.Store(id={"type": "data-report", "index": 7}),

                dbc.Col([
                    dbc.Row([
                        html.H4("Classifier performance metrics",
                                style={'margin-left': '1.5vw', 'margin-top': '2vw', 'margin-bottom': '1vw'},
                                id="title-dla")
                    ]),
                    dbc.Row(id="sec-dla-rf",
                            style={'margin-left': '2vw'}),
                    dbc.Row(id="sec-dla-knn",
                            style={'margin-left': '2vw'}),
                    dbc.Row(id="sec-dla-dt",
                            style={'margin-left': '2vw'}),
                    dbc.Row(id="sec-dla-svm",
                            style={'margin-left': '2vw'}),
                    dbc.Row(id="sec-dla-mlp",
                            style={'margin-left': '2vw'}),
                ], width={'size': 6}),

                # Boxplot results
                dbc.Col([html.Div(id="output-boxplot-dla")], width={'size': 6}, align='center'),

            ])),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


def classify_real_vs_synthetic_data(real, synthetic, numeric_features, categorical_features):
    real["label"] = 0
    synthetic["label"] = 1

    combined_data = pd.concat([real, synthetic], ignore_index=True)

    train_data, test_data, train_labels, test_labels = train_test_split(combined_data.drop("label", axis=1),
                                                                        combined_data["label"], test_size=0.2,
                                                                        random_state=9)

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ])

    train_data_preprocessed = preprocessor.fit_transform(train_data)
    test_data_preprocessed = preprocessor.transform(test_data)

    classifiers = {'RF': RandomForestClassifier(n_estimators=100, random_state=9, n_jobs=3),
                   'KNN': KNeighborsClassifier(n_neighbors=9, n_jobs=3),
                   'DT': DecisionTreeClassifier(random_state=9),
                   'SVM': SVC(C=100, max_iter=300, kernel='linear', probability=True, random_state=9),
                   'MLP': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=9)}

    results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1'])

    for clas_name in classifiers.keys():
        classifiers[clas_name].fit(train_data_preprocessed, train_labels)

        predictions = classifiers[clas_name].predict(test_data_preprocessed)

        clas_results = pd.DataFrame([[clas_name,
                                      np.round(accuracy_score(test_labels, predictions), 4),
                                      np.round(precision_score(test_labels, predictions), 4),
                                      np.round(recall_score(test_labels, predictions), 4),
                                      np.round(f1_score(test_labels, predictions), 4)]],
                                    columns=results.columns)

        results = pd.concat([results, clas_results], ignore_index=True)

    return results


@app.callback(Output('sec-dla-rf', 'children'),
              Output('sec-dla-knn', 'children'),
              Output('sec-dla-dt', 'children'),
              Output('sec-dla-svm', 'children'),
              Output('sec-dla-mlp', 'children'),
              Output({"type": "data-report", "index": 7}, 'data'),
              Input('title-dla', 'children'))
def fill_with_results(children_in):
    if children_in is not None:
        r = real_data.copy()
        s = synthetic_data.copy()
        res_dla = classify_real_vs_synthetic_data(r, s, num_features, cat_features)

        children_rf = [
            html.H5("Random Forest",
                    style={'margin-top': '2vw', 'margin-bottom': '1vw'}),
            dbc.Col(html.Div([
                html.B("Accuracy: "),
                html.P(res_dla[res_dla['model'] == "RF"]['accuracy'])
            ])),
            dbc.Col(html.Div([
                html.B("Precision: "),
                html.P(res_dla[res_dla['model'] == "RF"]['precision'])
            ])),
            dbc.Col(html.Div([
                html.B("Recall: "),
                html.P(res_dla[res_dla['model'] == "RF"]['recall'])
            ])),
            dbc.Col(html.Div([
                html.B("F1-score: "),
                html.P(res_dla[res_dla['model'] == "RF"]['f1'])
            ])),
        ]

        children_knn = [
            html.H5("K-Nearest Neighbors",
                    style={'margin-top': '2vw', 'margin-bottom': '1vw'}),
            dbc.Col(html.Div([
                html.B("Accuracy: "),
                html.P(res_dla[res_dla['model'] == "KNN"]['accuracy'])
            ])),
            dbc.Col(html.Div([
                html.B("Precision: "),
                html.P(res_dla[res_dla['model'] == "KNN"]['precision'])
            ])),
            dbc.Col(html.Div([
                html.B("Recall: "),
                html.P(res_dla[res_dla['model'] == "KNN"]['recall'])
            ])),
            dbc.Col(html.Div([
                html.B("F1-score: "),
                html.P(res_dla[res_dla['model'] == "KNN"]['f1'])
            ])),
        ]

        children_dt = [
            html.H5("Decision Tree",
                    style={'margin-top': '2vw', 'margin-bottom': '1vw'}),
            dbc.Col(html.Div([
                html.B("Accuracy: "),
                html.P(res_dla[res_dla['model'] == "DT"]['accuracy'])
            ])),
            dbc.Col(html.Div([
                html.B("Precision: "),
                html.P(res_dla[res_dla['model'] == "DT"]['precision'])
            ])),
            dbc.Col(html.Div([
                html.B("Recall: "),
                html.P(res_dla[res_dla['model'] == "DT"]['recall'])
            ])),
            dbc.Col(html.Div([
                html.B("F1-score: "),
                html.P(res_dla[res_dla['model'] == "DT"]['f1'])
            ])),
        ]

        children_svm = [
            html.H5("Support Vector Machines",
                    style={'margin-top': '2vw', 'margin-bottom': '1vw'}),
            dbc.Col(html.Div([
                html.B("Accuracy: "),
                html.P(res_dla[res_dla['model'] == "SVM"]['accuracy'])
            ])),
            dbc.Col(html.Div([
                html.B("Precision: "),
                html.P(res_dla[res_dla['model'] == "SVM"]['precision'])
            ])),
            dbc.Col(html.Div([
                html.B("Recall: "),
                html.P(res_dla[res_dla['model'] == "SVM"]['recall'])
            ])),
            dbc.Col(html.Div([
                html.B("F1-score: "),
                html.P(res_dla[res_dla['model'] == "SVM"]['f1'])
            ])),
        ]

        children_mlp = [
            html.H5("Multilayer Perceptron",
                    style={'margin-top': '2vw', 'margin-bottom': '1vw'}),
            dbc.Col(html.Div([
                html.B("Accuracy: "),
                html.P(res_dla[res_dla['model'] == "MLP"]['accuracy'])
            ])),
            dbc.Col(html.Div([
                html.B("Precision: "),
                html.P(res_dla[res_dla['model'] == "MLP"]['precision'])
            ])),
            dbc.Col(html.Div([
                html.B("Recall: "),
                html.P(res_dla[res_dla['model'] == "MLP"]['recall'])
            ])),
            dbc.Col(html.Div([
                html.B("F1-score: "),
                html.P(res_dla[res_dla['model'] == "MLP"]['f1'])
            ])),
        ]

        return children_rf, children_knn, children_dt, children_svm, children_mlp, res_dla.to_dict("list")


@app.callback(Output('output-boxplot-dla', 'children'),
              Input({"type": "data-report", "index": 7}, 'data'))
def update_graphs(data):
    if data is not None:
        fig = go.Figure()
        fig.add_trace(go.Box(y=data['accuracy'], name="Accuracy"))
        fig.add_trace(go.Box(y=data['precision'], name="Precision"))
        fig.add_trace(go.Box(y=data['recall'], name="Recall"))
        fig.add_trace(go.Box(y=data['f1'], name="F1-score"))

        children = [
            dcc.Graph(figure=fig)
        ]

        return children


# PAGE 3 CONTENTS (utility metrics)
page_3 = html.Div([

    dbc.Container(
        [
            # Header Utility section
            dbc.Row(
                dbc.Col([html.H4(["Utility Evaluation "],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")),

            # Upload train and test datasets
            dbc.Row([

                dbc.Col([
                    html.P(["Import datasets: ",
                            html.Span("ℹ", id="info-utl", style={'cursor': 'pointer'})]),
                    dbc.Tooltip(
                        "Import the training and testing datasets, otherwise, a random split will be performed on "
                        "the existing real dataset already imported.",
                        target="info-utl",
                    )
                ], width={'size': 2}),

                dbc.Col([
                    dcc.Upload([dbc.Button(["Upload Train"], id="button-train", color="primary")],
                               id="upload-data-train-utl",
                               multiple=False)
                ], width={'size': 'auto'}),

                dbc.Col([
                    dcc.Upload([dbc.Button(["Upload Test"], id="button-test", color="primary")],
                               id="upload-data-test-utl",
                               multiple=False),
                ], width={'size': 'auto'}),

            ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

            dbc.Modal(
                [
                    dbc.ModalHeader("Error"),
                    dbc.ModalBody("The uploaded train dataset does not have the same columns as the real dataset. "
                                  "Please load another file."),
                    dbc.ModalFooter(dbc.Button("Close", id="close1", className="ml-auto")),
                ],
                id="error-utl-train",
                centered=True,
                is_open=False,
            ),

            dbc.Modal(
                [
                    dbc.ModalHeader("Error"),
                    dbc.ModalBody("The uploaded test dataset does not have the same columns as the real dataset. "
                                  "Please load another file."),
                    dbc.ModalFooter(dbc.Button("Close", id="close2", className="ml-auto")),
                ],
                id="error-utl-test",
                centered=True,
                is_open=False,
            ),

            dbc.Form([
                # Dropdown for target class
                dbc.Row([
                    dbc.Col([
                        html.P("Select target class:")
                    ], width={'size': 2}),

                    dbc.Col([
                        dbc.Select(
                            id="dropdown-target",
                            options=[],
                            required='required'
                        )
                    ], width={'size': 2}),
                ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                # Dropdown for classifier
                dbc.Row([
                    dbc.Col([
                        html.P("Select classifier method:")
                    ], width={'size': 2}),

                    dbc.Col([
                        dbc.Select(
                            id="dropdown-classifier",
                            options=[
                                {"label": "Random Forest", "value": "RF"},
                                {"label": "K-Nearest Neighbors", "value": "KNN"},
                                {"label": "Decision Tree", "value": "DT"},
                                {"label": "Support Vector Machines", "value": "SVM"},
                                {"label": "Multilayer Perceptron", "value": "MLP"},
                            ],
                            required='required'
                        )
                    ], width={'size': 2}),
                ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                dbc.Row([
                    dbc.Col([dbc.Button("Evaluate Utility", color="info", id="run-utl")],
                            width={'size': 'auto'})
                ], style={'margin-bottom': '6vw', 'margin-left': '4vw'})
            ]),

            # Results section
            dcc.Loading(dbc.Row([
                dbc.Col(id="output-utl-trtr", width=5),
                dbc.Col(id="output-utl-tstr", width={'size': 5, 'offset': 1}),
            ], style={'margin-bottom': '0vw', 'margin-left': '4vw'})),
            dcc.Store(id={"type": "data-report", "index": 8}),

            dbc.Row(id="output-utl-diff", style={'margin-bottom': '0vw', 'margin-left': '4vw'}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('dropdown-target', 'options'),
              Input('dropdown-classifier', 'options'))
def update_option(op):
    if not real_data.empty:
        return [{'label': col, 'value': col} for col in cat_features]
    else:
        return []


@app.callback(Output('error-utl-train', 'is_open'),
              Output('button-train', 'color'),
              Input('upload-data-train-utl', 'contents'),
              Input('close1', 'n_clicks'),
              State('upload-data-train-utl', 'filename'))
def upload_train_dataset(contents, n_clicks, filename):
    global real_train_data

    ctx = dash.callback_context
    if not ctx.triggered:
        if not real_train_data.empty:
            return False, 'success'
        return False, 'primary'
    if ctx.triggered[0]['prop_id'] == 'close1.n_clicks':
        if not real_train_data.empty:
            return False, 'success'
        return False, 'primary'

    if contents is not None:
        real_train_data = parse_contents(contents, filename)

        global encoder_labels
        for col in real_data.columns:
            if real_train_data[col].dtype == 'object':
                real_train_data[col] = encoder_labels.transform(real_train_data[col])

        if list(real_data.columns) == list(real_train_data.columns):
            return False, 'success'
        else:
            real_train_data = pd.DataFrame()
            return True, 'primary'


@app.callback(Output('error-utl-test', 'is_open'),
              Output('button-test', 'color'),
              Input('upload-data-test-utl', 'contents'),
              Input('close2', 'n_clicks'),
              State('upload-data-test-utl', 'filename'))
def upload_test_dataset(contents, n_clicks, filename):
    global real_test_data

    ctx = dash.callback_context
    if not ctx.triggered:
        if not real_test_data.empty:
            return False, 'success'
        return False, 'primary'
    if ctx.triggered[0]['prop_id'] == 'close2.n_clicks':
        if not real_test_data.empty:
            return False, 'success'
        return False, 'primary'

    if contents is not None:
        real_test_data = parse_contents(contents, filename)

        global encoder_labels
        for col in real_data.columns:
            if real_test_data[col].dtype == 'object':
                real_test_data[col] = encoder_labels.transform(real_test_data[col])

        if list(real_data.columns) == list(real_test_data.columns):
            return False, 'success'
        else:
            real_test_data = pd.DataFrame()
            return True, 'primary'


def train_test_model(model_name, train_data, test_data, train_labels, test_labels, numeric_features,
                     categorical_features):
    models = {
        "RF": RandomForestClassifier(n_estimators=100, n_jobs=3, random_state=10),
        "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=3),
        "DT": DecisionTreeClassifier(random_state=10),
        "SVM": SVC(C=100, max_iter=300, kernel='linear', probability=True, random_state=10),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=10)
    }

    model = models[model_name]

    numeric_transformer = StandardScaler()
    data = pd.concat([train_data, test_data], ignore_index=True)
    categories_list = [np.unique(data[col]) for col in categorical_features]
    categorical_transformer = OneHotEncoder(categories=categories_list)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ])

    train_data_preprocessed = preprocessor.fit_transform(train_data)
    test_data_preprocessed = preprocessor.transform(test_data)

    model.fit(train_data_preprocessed, train_labels)
    predictions = model.predict(test_data_preprocessed)

    results = pd.DataFrame([[model_name,
                             np.round(accuracy_score(test_labels, predictions), 4),
                             np.round(precision_score(test_labels, predictions, average=None)[-1], 4),
                             np.round(recall_score(test_labels, predictions, average=None)[-1], 4),
                             np.round(f1_score(test_labels, predictions, average=None)[-1], 4)]],
                           columns=['model', 'accuracy', 'precision', 'recall', 'f1'])

    return results, confusion_matrix(test_labels, predictions)


@app.callback(Output('output-utl-trtr', 'children'),
              Output('output-utl-tstr', 'children'),
              Output('output-utl-diff', 'children'),
              Output({"type": "data-report", "index": 8}, 'data'),
              Input('run-utl', 'n_clicks'),
              [State('dropdown-target', 'value'),
               State('dropdown-classifier', 'value')])
def run_code_on_click(n_clicks, target, classifier):
    if n_clicks is not None and target is not None and classifier is not None:
        if not real_train_data.empty and not real_test_data.empty:
            train_data_r = real_train_data.drop(columns=target)
            test_data_r = real_test_data.drop(columns=target)
            train_labels_r = real_train_data[target]
            test_labels_r = real_test_data[target]
        else:
            train_data_r, test_data_r, train_labels_r, test_labels_r = train_test_split(
                real_data.drop(columns=target), real_data[target], test_size=0.2, random_state=9)

        num = num_features.copy()
        cat = cat_features.copy()
        if target in num:
            num.remove(target)

        if target in cat:
            cat.remove(target)

        results_TRTR, mat_trtr = train_test_model(classifier, train_data_r, test_data_r,
                                                  train_labels_r.astype(str), test_labels_r.astype(str),
                                                  num, cat)

        train_data_s, _, train_labels_s, _ = train_test_split(synthetic_data.drop(columns=target),
                                                              synthetic_data[target], test_size=0.2, random_state=19)

        results_TSTR, mat_tstr = train_test_model(classifier, train_data_s, test_data_r,
                                                  train_labels_s.astype(str), test_labels_r.astype(str),
                                                  num, cat)

        diff = np.round(abs(results_TRTR.drop('model', axis=1) - results_TSTR.drop('model', axis=1)), 4)

        fig_trtr = go.Figure(data=go.Heatmap(z=mat_trtr[::-1], text=mat_trtr[::-1],
                                             x=list(map(str, np.unique(synthetic_data[target]))),
                                             y=list(map(str, np.unique(synthetic_data[target])))[::-1],
                                             texttemplate="%{text}", colorscale='viridis', showscale=False))
        fig_trtr.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", title="Confusion Matrix")

        fig_tstr = go.Figure(data=go.Heatmap(z=mat_tstr[::-1], text=mat_tstr[::-1],
                                             x=list(map(str, np.unique(synthetic_data[target]))),
                                             y=list(map(str, np.unique(synthetic_data[target])))[::-1],
                                             texttemplate="%{text}", colorscale='viridis', showscale=False))
        fig_tstr.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", title="Confusion Matrix")

        children_TRTR = [
            dbc.Row([
                html.H4("Train on Real and Test on Real results"),
            ]),
            dbc.Row([
                dbc.Col([html.Div([
                    html.B("Accuracy: "),
                    html.P(results_TRTR['accuracy'])
                ])], width={'size': 2}),
                dbc.Col([html.Div([
                    html.B("Precision: "),
                    html.P(results_TRTR['precision'])
                ])], width={'size': 2, 'offset': 1}),
                dbc.Col([html.Div([
                    html.B("Recall: "),
                    html.P(results_TRTR['recall'])
                ])], width={'size': 2, 'offset': 1}),
                dbc.Col([html.Div([
                    html.B("F1-score: "),
                    html.P(results_TRTR['f1'])
                ])], width={'size': 2, 'offset': 1}),
            ]),
            dbc.Row([dbc.Col([
                dcc.Graph(figure=fig_trtr)
            ], width=9)], justify='center'),
        ]

        children_TSTR = [
            dbc.Row([
                html.H4("Train on Synthetic and Test on Real results"),
            ]),
            dbc.Row([
                dbc.Col([html.Div([
                    html.B("Accuracy: "),
                    html.P(results_TSTR['accuracy'])
                ])], width={'size': 2}),
                dbc.Col([html.Div([
                    html.B("Precision: "),
                    html.P(results_TSTR['precision'])
                ])], width={'size': 2, 'offset': 1}),
                dbc.Col([html.Div([
                    html.B("Recall: "),
                    html.P(results_TSTR['recall'])
                ])], width={'size': 2, 'offset': 1}),
                dbc.Col([html.Div([
                    html.B("F1-score: "),
                    html.P(results_TSTR['f1'])
                ])], width={'size': 2, 'offset': 1}),
            ]),
            dbc.Row([dbc.Col([
                dcc.Graph(figure=fig_tstr)
            ], width=9)], justify='center'),
        ]

        children_diff = [
            html.H4("Differences"),
            dbc.Col([html.Div([
                html.B("Accuracy: "),
                html.P(diff['accuracy'])
            ])], width={'size': 2}),
            dbc.Col([html.Div([
                html.B("Precision: "),
                html.P(diff['precision'])
            ])], width={'size': 2}),
            dbc.Col([html.Div([
                html.B("Recall: "),
                html.P(diff['recall'])
            ])], width={'size': 2}),
            dbc.Col([html.Div([
                html.B("F1-score: "),
                html.P(diff['f1'])
            ])], width={'size': 2}),
        ]

        return children_TRTR, children_TSTR, children_diff, \
               [results_TRTR.to_dict("list"), results_TSTR.to_dict("list"), fig_trtr, fig_tstr]
    else:
        return [], [], [], []


# PAGE 4 CONTENTS (privacy metrics)

# SEA metrics
page_4_sea = html.Div([

    dbc.Container(
        [
            # Header SEA section
            dbc.Row(
                dbc.Col(html.H4("Similarity Evaluation Analysis",
                                style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'}),
                        width="auto")),

            # Dropdown distance selection
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose metric similarity", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-sea",
                                options=[
                                    {"label": "Cosine similarity", "value": "cos"},
                                    {"label": "Euclidean distance", "value": "euc"},
                                    {"label": "Hausdorff distance", "value": "hau"},
                                ],
                                value="cos",
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                ]
            ),

            # Graph distances
            dcc.Loading(dbc.Row(id="output-graph-sea"), id={"type": "load-res", "index": 2}),
            dcc.Store(id={"type": "data-report", "index": 9}),

            # dbc.Row(id="output-box-select-sea"),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


def pairwise_euclidean_distance(real, synthetic):
    distances = distance.cdist(real, synthetic, 'euclidean')

    return np.round(distances, 4)


def str_similarity(real, synthetic):
    distances = cosine_similarity(real, synthetic)

    return np.round(distances, 4)


def hausdorff_distance(real, synthetic):
    distances = max(distance.directed_hausdorff(real, synthetic)[0], distance.directed_hausdorff(synthetic, real)[0])

    return np.round(distances, 4)


@app.callback(Output('output-graph-sea', 'children'),
              Output({"type": "data-report", "index": 9}, 'data'),
              Input('dropdown-sea', 'value'))
def update_graph(value):
    if value is not None:

        if value == "euc":
            mat_dist = pairwise_euclidean_distance(scale_data(real_data), scale_data(synthetic_data))
        elif value == "cos":
            mat_dist = str_similarity(real_data, synthetic_data)
        elif value == "hau":
            dist = hausdorff_distance(scale_data(real_data), scale_data(synthetic_data))

        if value == "euc" or value == "cos":
            fig_box = ff.create_distplot([mat_dist.flatten()], ['Paired distance values'],
                                         show_hist=False, show_rug=False, colors=['blue'])
            fig_box.update_traces(fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.5)', selector=dict(type='scatter'))

            children = [
                dbc.Col(html.Div([dcc.Graph(figure=fig_box, id="box-interactive-sea")]),
                        width={'size': 10, 'offset': 1}, align="center"),
            ]
        else:
            fig_box = dist
            children = [
                dbc.Col(html.Div([html.B("Hausdorff distance between synthetic and real datasets: "), html.P(dist)]),
                        width={'size': 6, 'offset': 1}, align="center"),
            ]

        return children, [fig_box, value]
    else:
        return [], []


# MIA metrics
page_4_mia = html.Div([
    dbc.Container(
        [

            # Header MIA section
            dbc.Row([
                dbc.Col([html.H4(["Membership Inference Attack"],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")
            ]),

            dbc.Row([

                # Attack schema
                dbc.Col([
                    dbc.Toast(
                        [
                            html.Img(
                                src="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10306449/bin/10-1055-s-0042-1760247-i22020006-2.jpg",
                                style={'width': '100%', 'height': 'auto',
                                       'borderRadius': '10px', 'boxShadow': '2px 2px 5px grey'
                                       }
                            )
                        ],
                        style={'width': '85%'},
                        header="MIA schema",
                    )
                ], width={'size': 6}),

                dbc.Col([

                    # Upload training set
                    dbc.Row([
                        dbc.Col([
                            html.P("Import training dataset:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Upload([dbc.Button(["Upload Train"], id="button-train-pr", color="primary")],
                                       id="upload-data-train-pr",
                                       multiple=False)
                        ], width={'size': 'auto'}),
                    ], style={'margin-bottom': '2vw'}),

                    dbc.Modal(
                        [
                            dbc.ModalHeader("Error"),
                            dbc.ModalBody(
                                "The uploaded train dataset does not have the same columns as the real dataset. "
                                "Please load another file."),
                            dbc.ModalFooter(dbc.Button("Close", id="close3", className="ml-auto")),
                        ],
                        id="error-pr-train",
                        centered=True,
                        is_open=False,
                    ),

                    dbc.Form([
                        # Slider subset
                        dbc.Row([
                            dbc.Col([
                                html.P("Enter subset size:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Slider(10, 100, 5,
                                           tooltip={"placement": "bottom", "always_visible": True},
                                           marks={
                                               10: '10%',
                                               25: '25%',
                                               50: '50%',
                                               75: '75%',
                                               100: '100%'
                                           },
                                           value=40,
                                           id="slider-subset")
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        # Slider similarità
                        dbc.Row([
                            dbc.Col([
                                html.P("Enter similarity threshold:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Slider(0, 1, 0.05,
                                           tooltip={"placement": "bottom", "always_visible": True},
                                           marks={
                                               0: 'Low',
                                               1: 'High'
                                           },
                                           value=0.6,
                                           id="slider-sim")
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        dbc.Row([
                            dbc.Col([dbc.Button("Simulate Attack", color="info", id="run-mia")],
                                    width={'size': 'auto'})
                        ], style={'margin-bottom': '0vw'})
                    ]),

                    # Graphs performance attacker
                    dcc.Loading(dbc.Row(html.Div(id="output-mia"))),
                    dcc.Store(id={"type": "data-report", "index": 10})

                ], width={'size': 5}, align='center')

            ], style={'margin-left': '4vw'}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('error-pr-train', 'is_open'),
              Output('button-train-pr', 'color'),
              Input('upload-data-train-pr', 'contents'),
              Input('close3', 'n_clicks'),
              State('upload-data-train-pr', 'filename'))
def upload_train_dataset(contents, n_clicks, filename):
    global real_train_data
    ctx = dash.callback_context
    if not ctx.triggered:
        if not real_train_data.empty:
            return False, 'success'
        return False, 'primary'
    if ctx.triggered[0]['prop_id'] == 'close3.n_clicks':
        if not real_train_data.empty:
            return False, 'success'
        return False, 'primary'

    if contents is not None:
        real_train_data = parse_contents(contents, filename)

        if list(real_data.columns) == list(real_train_data.columns):
            return False, 'success'
        else:
            real_train_data = pd.DataFrame()
            return True, 'primary'


@app.callback(Output('run-mia', 'disabled'),
              Input('button-train-pr', 'color'))
def enable_submit(color):
    if color == 'success':
        return False
    else:
        return True


def membership_inference_attack(real_subset_attacker, label_membership_train, synthetic, threshold):
    distances = cosine_similarity(real_subset_attacker, synthetic)

    records_identified = (distances > threshold).any(axis=1)  # if TRUE, the real row is identified

    precision_attacker = precision_score(label_membership_train, records_identified)
    accuracy_attacker = accuracy_score(label_membership_train, records_identified)

    return precision_attacker, accuracy_attacker


@app.callback(Output('output-mia', 'children'),
              Output({"type": "data-report", "index": 10}, 'data'),
              Input('run-mia', 'n_clicks'),
              State('slider-subset', 'value'),
              State('slider-sim', 'value'))
def run_code_on_click(click, prop_subset, t_similarity):
    if click:

        real_subset = real_data.sample(frac=prop_subset / 100, random_state=42).reset_index(drop=True)

        label_membership_train = [tuple(row) in set(real_train_data.itertuples(index=False))
                                  for row in real_subset.itertuples(index=False)]

        precision_attacker, accuracy_attacker = membership_inference_attack(real_subset, label_membership_train,
                                                                            synthetic_data, t_similarity)

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
        fig.add_trace(go.Pie(labels=['Correct', 'Wrong'], values=[accuracy_attacker, 1 - accuracy_attacker]), 1, 1)
        fig.add_trace(go.Pie(labels=['Correct', 'Wrong'], values=[precision_attacker, 1 - precision_attacker]), 1, 2)

        fig.update_traces(hole=.4, marker=dict(colors=['lightgreen', 'black']))

        fig.update_layout(annotations=[dict(text='Accuracy', xref="x domain", yref="y domain", x=0.14, y=0.5,
                                            font_size=15, showarrow=False),
                                       dict(text='Precision', xref="x domain", yref="y domain", x=0.86, y=0.5,
                                            font_size=15, showarrow=False)],
                          showlegend=False,
                          title="Attacker performance")

        children = [
            dcc.Graph(figure=fig)
        ]

        return children, [{'precision': precision_attacker, 'accuracy': accuracy_attacker}, fig, prop_subset,
                          t_similarity]
    else:
        return [], []


# AIA metrics
page_4_aia = html.Div([
    dbc.Container(
        [

            # Header AIA section
            dbc.Row([
                dbc.Col([html.H4(["Attribute Inference Attack"],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")
            ]),

            dbc.Row([

                # Attack schema
                dbc.Col([
                    dbc.Toast(
                        [
                            html.Img(
                                src="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10306449/bin/10-1055-s-0042-1760247-i22020006-3.jpg",
                                style={'width': '100%', 'height': 'auto',
                                       'borderRadius': '10px', 'boxShadow': '2px 2px 5px grey'
                                       }
                            )
                        ],
                        style={'width': '85%'},
                        header="AIA schema",
                    )
                ], width={'size': 6}),

                dbc.Col([

                    dbc.Form([
                        # Slider subset size
                        dbc.Row([
                            dbc.Col([
                                html.P("Enter subset size:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Slider(10, 100, 5,
                                           tooltip={"placement": "bottom", "always_visible": True},
                                           marks={
                                               10: '10%',
                                               25: '25%',
                                               50: '50%',
                                               75: '75%',
                                               100: '100%'
                                           },
                                           value=40,
                                           id="slider-subset-aia")
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        # Dropdown for attributes selection
                        dbc.Row([
                            dbc.Col([
                                html.P("Attacker attributes:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Dropdown(
                                    options=[],
                                    multi=True,
                                    id="dropdown-aia"
                                )
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        dbc.Row([
                            dbc.Col([dbc.Button("Simulate Attack", color="info", id="run-aia")],
                                    width={'size': 'auto'})
                        ], style={'margin-bottom': '4vw'})
                    ]),

                    # Performance attacker
                    dcc.Loading(dbc.Row(html.Div(id="output-aia"))),
                    dcc.Store(id={"type": "data-report", "index": 11})

                ], width={'size': 5}, align='center')

            ], style={'margin-left': '4vw'}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('dropdown-aia', 'options'),
              Input('slider-subset-aia', 'marks'))
def update_options(m):
    if not real_data.empty:
        return [{'label': col, 'value': col} for col in real_data.columns]
    else:
        return []


@app.callback(Output('run-aia', 'disabled'),
              Input('dropdown-aia', 'value'))
def enable_submit(value):
    if value is not None and value:
        return False
    else:
        return True


def attribute_inference_attack(real, synthetic, QID_features_names, target_features_names, dict_type):
    real_subset_attacker = real[QID_features_names]

    train_synthetic_data_QID = synthetic[QID_features_names]

    train_features_type = {key: dict_type[key] for key in QID_features_names}

    train_numeric_features = [key for key, value in train_features_type.items() if value == "numerical"]
    train_categorical_features = [key for key, value in train_features_type.items() if value == "categorical"]

    numeric_transformer = StandardScaler()
    data = pd.concat([train_synthetic_data_QID, real_subset_attacker], ignore_index=True)
    categories_list = [np.unique(data[col]) for col in train_categorical_features]
    categorical_transformer = OneHotEncoder(categories=categories_list)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, train_numeric_features),
            ("categorical", categorical_transformer, train_categorical_features)
        ])

    train_data_preprocessed = preprocessor.fit_transform(train_synthetic_data_QID)
    test_data_preprocessed = preprocessor.transform(real_subset_attacker)

    results = list()

    for target_name in target_features_names:

        if dict_features_type[target_name] == "numerical":

            model = DecisionTreeRegressor(random_state=23)
            model.fit(train_data_preprocessed, synthetic[target_name])
            predictions = model.predict(test_data_preprocessed)
            results.append(
                ['rmse', target_name, np.round(mean_squared_error(real[target_name], predictions, squared=False), 4),
                 str([np.round(np.percentile(real[target_name], 25), 4),
                      np.round(np.percentile(real[target_name], 75), 4)])])

        else:

            model = DecisionTreeClassifier(random_state=23)
            model.fit(train_data_preprocessed, synthetic[target_name].astype(str))
            predictions = model.predict(test_data_preprocessed)
            results.append(
                ['acc', target_name, np.round(accuracy_score(real[target_name].astype(str), predictions), 4), []])

    return pd.DataFrame(results, columns=['Metric name', 'Target name', 'Value', 'IQR target'])


@app.callback(Output('output-aia', 'children'),
              Output({"type": "data-report", "index": 11}, 'data'),
              Input('run-aia', 'n_clicks'),
              State('slider-subset-aia', 'value'),
              State('dropdown-aia', 'value'))
def run_code_on_click(click, prop_subset, attributes):
    if click:

        real_subset = real_data.sample(frac=prop_subset / 100, random_state=24).reset_index(drop=True)
        targets = [col for col in real_data.columns if col not in attributes]

        results_aia = attribute_inference_attack(real_subset, synthetic_data, attributes, targets, dict_features_type)

        df_acc = results_aia[results_aia['Metric name'] == 'acc'][['Target name', 'Value']]
        df_rmse = results_aia[results_aia['Metric name'] == 'rmse'][['Target name', 'IQR target', 'Value']]

        children = [
            dbc.Tabs([
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                data=df_acc.to_dict('records'),
                                columns=[{'name': "Target name", 'id': "Target name"},
                                         {'name': "Value", 'id': "Value"}],
                                page_action='none',
                                style_table={'height': '300px', 'overflowY': 'auto'},
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                            )
                        ])
                    ], color="primary", outline=True)
                ], label="Accuracy", tab_id="tab1"),
                dbc.Tab([
                    dbc.Card([
                        dbc.CardBody([
                            dash_table.DataTable(
                                data=df_rmse.to_dict('records'),
                                columns=[{'name': "Target name", 'id': "Target name"},
                                         {'name': "IQR target", 'id': "IQR target"},
                                         {'name': "Value", 'id': "Value"}],
                                page_action='none',
                                style_table={'height': '300px', 'overflowY': 'auto'},
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': '33%', 'maxWidth': '33%', 'width': '33%'},
                            )
                        ])
                    ], color="primary", outline=True)
                ], label="RMSE", tab_id="tab2")
            ], active_tab="tab1")
        ]

        return children, [df_acc.to_dict('list'), df_rmse.to_dict('list'), prop_subset, attributes]

    else:
        return [], []


# REPORT SECTION
@app.callback(Output("download-report", "data"),
              Input("button-report", "n_clicks"),
              [State("user_position", "pathname"),
               State({"type": "data-report", "index": ALL}, "data")],
              prevent_initial_call=True)
def generate_and_download_report(n_clicks, pathname, data_report):
    if n_clicks is None:
        raise PreventUpdate

    if pathname == '/page_2_ura':
        data_ura_test_num, data_ura_test_cat, data_ura_dist = [data_report[i][0] for i in range(3)]
        selected_test_num, selected_test_cat, selected_dist = [data_report[i][1]['selected_opt'] for i in range(3)]

        if selected_test_num == 'ks_test':
            selected_test_num = 'Kolmogorov–Smirnov test'
        elif selected_test_num == 't_test':
            selected_test_num = 'Student t-test'
        elif selected_test_num == 'u_test':
            selected_test_num = 'Mann–Whitney U test'

        if selected_test_cat == 'chi_test':
            selected_test_cat = 'Chi-square test'

        if selected_dist == 'cos_dist':
            selected_dist = 'Cosine distance'
        elif selected_dist == 'js_dist':
            selected_dist = 'Jensen-Shannon distance'
        elif selected_dist == 'w_dist':
            selected_dist = 'Wasserstein distance'

        df_test_num = pd.DataFrame(list(data_ura_test_num.items()), columns=['Feature', 'p value'])
        df_test_cat = pd.DataFrame(list(data_ura_test_cat.items()), columns=['Feature', 'p value'])
        df_dist = pd.DataFrame(list(data_ura_dist.items()), columns=['Feature', 'Distance value'])

        df_test_num = \
            df_test_num.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                     '<table class="table table-striped">')
        df_test_cat = \
            df_test_cat.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                     '<table class="table table-striped">')
        df_dist = \
            df_dist.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                 '<table class="table table-striped">')

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{ margin:0 100; background:whitesmoke; }
                </style>
            </head>
            <body>
                <h1>Univariate Resemblance Analysis</h1>

                <h3>Numerical test results</h3>
                <h4>Statistical test selected: ''' + selected_test_num + '''</h4>
                ''' + df_test_num + '''
                </br>

                <h3>Categorical test results</h3>
                <h4>Statistical test selected: ''' + selected_test_cat + '''</h4>
                ''' + df_test_cat + '''
                </br>

                <h3>Distance metrics results</h3>
                <h4>Distance metric selected: ''' + selected_dist + '''</h4>
                ''' + df_dist + '''
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Resemblance URA.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Resemblance URA.html')
        converter.convert(f'file:///{path}', 'data_report/Report Resemblance URA.pdf')

        return dcc.send_file('data_report/Report Resemblance URA.pdf')

    elif pathname == '/page_2_mra':
        data_mra_corr = data_report[0]
        data_mra_lof = data_report[1]
        data_mra_pca = data_report[2]
        data_mra_umap = data_report[3]

        if data_mra_corr[1] == 'corr_num':
            corr_type = 'Pairwise Pearson correlation matrices'
        else:
            corr_type = 'Normalized contingency tables'

        if data_mra_corr[2] == 'rs':
            vis_corr_type = 'Real data vs. Synthetic data'
            path_corr_r = os.path.join(path_user, 'data_figures', 'mat_corr_r.html').replace("\\", '/')
            pio.write_html(data_mra_corr[0][0], file=path_corr_r, auto_open=False)
            path_corr_s = os.path.join(path_user, 'data_figures', 'mat_corr_s.html').replace("\\", '/')
            pio.write_html(data_mra_corr[0][1], file=path_corr_s, auto_open=False)
            html_corr = '''
                <div style="display: flex;">
                    <div style="width: 800;">
                        <iframe width="400" height="400" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                    </div>
                    <div style="width: 40%;">
                        <iframe width="400" height="400" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                    </div>
                </div>
            '''.format(path_corr_r, path_corr_s)
        else:
            vis_corr_type = 'Differences between Real data and Synthetic data'
            path_corr_diff = os.path.join(path_user, 'data_figures', 'mat_corr_d.html').replace("\\", '/')
            pio.write_html(data_mra_corr[0], file=path_corr_diff, auto_open=False)
            html_corr = '<iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no"' \
                        ' src="' + path_corr_diff + '">' \
                                                    '</iframe></br>'

        path_lof = os.path.join(path_user, 'data_figures', 'box_lof.html').replace("\\", '/')
        pio.write_html(data_mra_lof, file=path_lof, auto_open=False)

        diff = np.round(abs(np.array(data_mra_pca[0]['yr']) - np.array(data_mra_pca[0]['ys'])), 2)
        df_pca = pd.DataFrame(list(zip(data_mra_pca[0]['x'], diff)), columns=['Component', 'Difference (%)'])

        df_pca = \
            df_pca.to_html(index=False). \
                replace('<table border="1" class="dataframe">', '<table class="table table-striped">')

        path_pca = os.path.join(path_user, 'data_figures', 'pca.html').replace("\\", '/')
        pio.write_html(data_mra_pca[1], file=path_pca, auto_open=False)

        path_umap = os.path.join(path_user, 'data_figures', 'umap.html').replace("\\", '/')
        pio.write_html(data_mra_umap[0], file=path_umap, auto_open=False)

        if data_mra_umap[3] == 'rs':
            vis_type = 'Real data vs. Synthetic data'
        else:
            vis_type = 'Real data together with Synthetic data'

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:whitesmoke; }</style>
            </head>
            <body>
                <h1>Multivariate Relationship Analysis</h1>

                <h3>Correlation matrices results</h3>
                <h4 style="display: list-item;">Matrix type selected: ''' + corr_type + '''</h4>
                <h4 style="display: list-item;">Visualization mode: ''' + vis_corr_type + '''</h4>                
                ''' + html_corr + '''

                <h3>Local Outlier Factor results</h3>
                <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path_lof + '''"></iframe>
                </br>

                <h3>Principal Components Analysis results</h3>
                <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path_pca + '''"></iframe>
                </br>
                ''' + df_pca + '''
                </br>

                <h3>UMAP results</h3>
                <h4 style="display: list-item;">Number of neighbors: ''' + str(data_mra_umap[1]) + '''</h4>
                <h4 style="display: list-item;">Minimum distance: ''' + str(data_mra_umap[2]) + '''</h4>
                <h4 style="display: list-item;">Visualization mode: ''' + vis_type + '''</h4>                
                <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path_umap + '''"></iframe>
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Resemblance MRA.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Resemblance MRA.html')
        converter.convert(f'file:///{path}', 'data_report/Report Resemblance MRA.pdf')

        return dcc.send_file('data_report/Report Resemblance MRA.pdf')

    elif pathname == '/page_2_dla':
        data_dla = data_report[0]

        df_dla = pd.DataFrame(data_dla)
        df_dla.columns = ['Name model', 'Accuracy', 'Precision', 'Recall', 'F1 score']
        df_dla['Name model'] = ['Random Forest', 'K-Nearest Neighbors', 'Decision Tree', 'Support Vector Machine',
                                'Multilayer Perceptron']

        df_dla = \
            df_dla.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                '<table class="table table-striped">')

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{ margin:0 100; background:whitesmoke; }
                </style>
            </head>
            <body>
                <h1>Data Labeling Analysis</h1>

                <h3>Performance metrics results</h3>
                ''' + df_dla + '''
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Resemblance DLA.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Resemblance DLA.html')
        converter.convert(f'file:///{path}', 'data_report/Report Resemblance DLA.pdf')

        return dcc.send_file('data_report/Report Resemblance DLA.pdf')

    elif pathname == '/page_3':
        data_trtr = data_report[0][0]
        data_tstr = data_report[0][1]
        fig_trtr = data_report[0][2]
        fig_tstr = data_report[0][3]

        df_trtr = pd.DataFrame(data_trtr)
        df_tstr = pd.DataFrame(data_tstr)
        df_trtr.columns = ['Name model', 'Accuracy', 'Precision', 'Recall', 'F1 score']
        df_tstr.columns = ['Name model', 'Accuracy', 'Precision', 'Recall', 'F1 score']
        name_model = df_trtr['Name model'].iloc[0]

        if name_model == 'RF':
            df_trtr['Name model'] = 'Random Forest'
            df_tstr['Name model'] = 'Random Forest'
        elif name_model == 'KNN':
            df_trtr['Name model'] = 'K-Nearest Neighbors'
            df_tstr['Name model'] = 'K-Nearest Neighbors'
        elif name_model == 'DT':
            df_trtr['Name model'] = 'Decision Tree'
            df_tstr['Name model'] = 'Decision Tree'
        elif name_model == 'SVM':
            df_trtr['Name model'] = 'Support Vector Machine'
            df_tstr['Name model'] = 'Support Vector Machine'
        elif name_model == 'MLP':
            df_trtr['Name model'] = 'Multilayer Perceptron'
            df_tstr['Name model'] = 'Multilayer Perceptron'

        df_trtr = \
            df_trtr.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                 '<table class="table table-striped">')
        df_tstr = \
            df_tstr.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                 '<table class="table table-striped">')

        path_trtr = os.path.join(path_user, 'data_figures', 'mat_trtr.html').replace("\\", '/')
        pio.write_html(fig_trtr, file=path_trtr, auto_open=False)

        path_tstr = os.path.join(path_user, 'data_figures', 'mat_tstr.html').replace("\\", '/')
        pio.write_html(fig_tstr, file=path_tstr, auto_open=False)

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{ margin:0 100; background:whitesmoke; }
                </style>
            </head>
            <body>
                <h1>Utility Evaluation</h1>

                <h3>Train on Real, Test on Real performance metrics results</h3>
                ''' + df_trtr + '''
                <iframe width="500" height="300" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path_trtr + '''"></iframe>
                </br>

                <h3>Train on Synthetic, Test on Real performance metrics results</h3>
                ''' + df_tstr + '''
                <iframe width="500" height="300" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path_tstr + '''"></iframe>
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Utility.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Utility.html')
        converter.convert(f'file:///{path}', 'data_report/Report Utility.pdf')

        return dcc.send_file('data_report/Report Utility.pdf')

    elif pathname == '/page_4_sea':
        data_fig_sea = data_report[0][0]
        selected_dist = data_report[0][1]

        if selected_dist == 'cos':
            selected_dist = 'Cosine similarity'
            path = os.path.join(path_user, 'data_figures', 'fig_sea.html').replace("\\", '/')
            pio.write_html(data_fig_sea, file=path, auto_open=False)
            html_dist = '''
                <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path + '''"></iframe>
            '''
        elif selected_dist == 'euc':
            selected_dist = 'Euclidean distance'
            path = os.path.join(path_user, 'data_figures', 'fig_sea.html').replace("\\", '/')
            pio.write_html(data_fig_sea, file=path, auto_open=False)
            html_dist = '''
                <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path + '''"></iframe>
            '''
        elif selected_dist == 'hau':
            selected_dist = 'Hausdorff distance'
            html_dist = '''
                <h4>Distance metric value: ''' + str(data_fig_sea) + '''</h4>
            '''

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{ margin:0 100; background:whitesmoke; }
                </style>
            </head>
            <body>
                <h1>Similarity Evaluation Analysis</h1>

                <h3>Paired distances results</h3>
                <h4>Distance metric selected: ''' + selected_dist + '''</h4>
                ''' + html_dist + '''
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Privacy SEA.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Privacy SEA.html')
        converter.convert(f'file:///{path}', 'data_report/Report Privacy SEA.pdf')

        return dcc.send_file('data_report/Report Privacy SEA.pdf')

    elif pathname == '/page_4_mia':
        data_performance_mia = data_report[0][0]
        data_fig_mia = data_report[0][1]
        prop_subset = data_report[0][2]
        t_sim = data_report[0][3]

        path = os.path.join(path_user, 'data_figures', 'pie_mia.html').replace("\\", '/')
        pio.write_html(data_fig_mia, file=path, auto_open=False)

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:whitesmoke; }</style>
            </head>
            <body>
                <h1>Membership Inference Attack</h1>

                <h3>Attacker information</h3>
                <h4>Real subset size: ''' + str(prop_subset) + '''% of the real dataset</h4>
                <h4>Similarity threshold: ''' + str(t_sim) + ''' (range 0-1)</h4>
                </br>

                <h3>Performance attacker results</h3>
                <h4>Precision: ''' + str(np.round(data_performance_mia['precision'], 4)) + '''</h4>
                <h4>Accuracy: ''' + str(np.round(data_performance_mia['accuracy'], 4)) + '''</h4>
                </br>
                <iframe width="800" height="400" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path + '''"></iframe>
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Privacy MIA.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Privacy MIA.html')
        converter.convert(f'file:///{path}', 'data_report/Report Privacy MIA.pdf')

        return dcc.send_file('data_report/Report Privacy MIA.pdf')

    elif pathname == '/page_4_aia':
        data_aia_acc = data_report[0][0]
        data_aia_rmse = data_report[0][1]
        prop_subset = data_report[0][2]
        attributes = data_report[0][3]

        df_acc = pd.DataFrame(data_aia_acc)
        df_rmse = pd.DataFrame(data_aia_rmse)
        df_acc.columns = ['Attribute re-identified', 'Attacker accuracy']
        df_rmse.columns = ['Attribute re-identified', 'Interquartile range', 'Attacker RMSE']

        df_acc = \
            df_acc.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                '<table class="table table-striped">')
        df_rmse = \
            df_rmse.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                 '<table class="table table-striped">')

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:whitesmoke; }</style>
            </head>
            <body>
                <h1>Attribute Inference Attack</h1>

                <h3>Attacker information</h3>
                <h4>Real subset size: ''' + str(prop_subset) + '''% of the real dataset</h4>
                <h4>Available attributes: ''' + str(attributes) + '''</h4>
                </br>

                <h3>Performance attacker results</h3>
                </br>
                <h4>Categorical attributes results</h4>
                ''' + df_acc + '''
                </br>
                <h4>Numerical attributes results</h4>
                ''' + df_rmse + '''
                </br>

            </body>
        </html>'''

        file_path = 'data_report/Report Privacy AIA.html'
        with open(file_path, 'w') as f:
            f.write(html_string)

        path = os.path.abspath('data_report/Report Privacy AIA.html')
        converter.convert(f'file:///{path}', 'data_report/Report Privacy AIA.pdf')

        return dcc.send_file('data_report/Report Privacy AIA.pdf')


@app.callback(Output("button-report", "disabled", allow_duplicate=True),
              Input("user_position", "pathname"),
              prevent_initial_call=True)
def download_disabled_change_position(p):
    return True


@app.callback(Output("button-report", "disabled"),
              Input({"type": "data-report", "index": ALL}, "data"),
              prevent_initial_call=True)
def update_download_disabled(data):
    return any(not obj or obj is None for obj in data)


if __name__ == '__main__':
    app.run_server(debug=False, host='127.0.0.1')
