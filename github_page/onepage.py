import json

import dash_bootstrap_components as dbc
import dash_renderjson
from dash import Dash, html, State, Output, Input, ClientsideFunction, dcc


def get_navbar():
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(
                                src="https://raw.githubusercontent.com/Open-Reaction-Database/ord-schema/main/logos/logo.svg",
                                height="30px")),
                            dbc.Col(dbc.NavbarBrand("Organic Synthesis Parser"), className="ms-2"),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    # href="https://open-reaction-database.org/",
                    style={"textDecoration": "none"},
                ),
            ],
        ),
        color="#C0C0C0",
        dark=False,
        className="mb-3",
        style=BAR_STYLE,
    )
    return navbar


def get_json_viewer_card(viewer_cid: str, header: str, data: dict | list):
    jv = dash_renderjson.DashRenderjson(id=viewer_cid, max_depth=-1, theme=JsonTheme, invert_theme=True, data=data)
    card = dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(jv)
        ]
    )
    return card


with open("test_results.json", "r") as f:
    test_results = json.load(f)

DATA = dict()
for d in test_results:
    reaction_id = d['data']['reaction_id']
    input_text = d['data']['input_text']
    truth = d['data']['output_reaction_inputs']
    try:
        infer = json.loads(d['completion'])['output_reaction_inputs']
    except:
        infer = None
    link = f'https://open-reaction-database.org/client/id/{reaction_id}'
    DATA[reaction_id] = [input_text, truth, infer, link]

reaction_id_0 = sorted(DATA.keys())[42]

JsonTheme = {
    "scheme": "monokai",
    "author": "wimer hazenberg (http://www.monokai.nl)",
    "base00": "#272822",
    "base01": "#383830",
    "base02": "#49483e",
    "base03": "#75715e",
    "base04": "#a59f85",
    "base05": "#f8f8f2",
    "base06": "#f5f4f1",
    "base07": "#f9f8f5",
    "base08": "#f92672",
    "base09": "#fd971f",
    "base0A": "#f4bf75",
    "base0B": "#a6e22e",
    "base0C": "#a1efe4",
    "base0D": "#66d9ef",
    "base0E": "#ae81ff",
    "base0F": "#cc6633",
}

CONTENT_STYLE = {
    "marginLeft": "2rem",
    "marginRight": "2rem",
    "padding": "1rem 1rem",
    "zIndex": "0",
}

BAR_STYLE = {
    "zIndex": "1",
}

app = Dash(
    name=__name__,
    title="Synthesis Parser",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

page_content = html.Div(
    [
        dbc.Card(
            [
                dbc.CardHeader("Text"),
                dbc.CardBody(
                    [
                        dcc.Dropdown(
                            id="reaction_id_dropdown",
                            value=reaction_id_0,
                            options=sorted(DATA.keys()),
                            className="mb-3"
                        ),
                        html.Div(DATA[reaction_id_0][0], id="reaction_text"),
                        html.Div(
                            html.A("ORD link", id="ord_link",
                                   href=f'https://open-reaction-database.org/client/id/{reaction_id_0}'),
                        )
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                html.Div(
                    get_json_viewer_card(
                        viewer_cid="parser-ref-output_reaction_inputs",
                        header="Ground truth",
                        data=DATA[reaction_id_0][1],
                    ),
                    className="col-6"
                ),
                html.Div(
                    get_json_viewer_card(
                        viewer_cid="parser-output-output_reaction_inputs",
                        header="LLM inferred",
                        data=DATA[reaction_id_0][2],
                    ),
                    className="col-6"
                ),
            ],
            className="mt-3"
        ),
    ]
)

content = html.Div(id="page-content", children=[page_content], style=CONTENT_STYLE)

navbar = get_navbar()

app.layout = html.Div(
    [
        dcc.Store(data=DATA, id="store-data"),
        navbar,
        content,
    ],
    style={"width": "100vw"}
)

app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_reaction'
    ),
    Output('reaction_text', 'children'),
    Output("parser-ref-output_reaction_inputs", "data"),
    Output("parser-output-output_reaction_inputs", "data"),
    Output("ord_link", "href"),
    Input('reaction_id_dropdown', 'value'),
    State('store-data', 'data')
)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8050, debug=False)
