import json
import os.path
import random

import dash_bootstrap_components as dbc
import dash_renderjson
import openai
from dash import html, get_app, register_page, Input, Output, State, dcc, ctx

random.seed(42)


def get_completion(prompt):
    prompt += "\n\n###\n\n"
    inference = openai.Completion.create(
        # model="curie:ft-llm-hackathon-synthesis-parsing-team-2023-03-29-22-57-05",
        model="davinci:ft-llm-hackathon-synthesis-parsing-team-2023-03-30-00-54-58",
        prompt=prompt,
        max_tokens=1000,
        stop=["###"]
    )
    completion_text = inference["choices"][0]["text"]
    return completion_text


register_page(__name__, path='/parser', description="Parser")

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

app = get_app()
this_dir = os.path.dirname(__file__)
train_data_path = os.path.join(this_dir, "../assets/data1000_v2.json")
test_data_path = os.path.join(this_dir, "../assets/data50_v2_test.json")

with open(train_data_path, "r") as f:
    train_data = json.load(f)

with open(test_data_path, "r") as f:
    test_data = json.load(f)

train_data_ids = set([r['reaction_id'] for r in train_data])


def get_json_viewer_card(viewer_cid: str, header: str, data_string: str = None):
    if data_string is None:
        jv = dash_renderjson.DashRenderjson(id=viewer_cid, max_depth=-1, theme=JsonTheme, invert_theme=True, )
    else:
        jv = dash_renderjson.DashRenderjson(id=viewer_cid, max_depth=-1, theme=JsonTheme, invert_theme=True,
                                            data=data_string)

    card = dbc.Card(
        [
            dbc.CardHeader(header),
            dbc.CardBody(jv)
        ]
    )
    return card


layout = html.Div(
    [
        html.H3([
            "Synthesis Parser",
        ]),
        html.Hr(),
        html.Div(
            [
                dbc.Button("random synthesis text [train]", id="parser-load-button-train", className="me-3"),
                dbc.Button("random synthesis text [test]", id="parser-load-button-test", className="me-3"),
                dbc.Button("INFER", id="parser-load-button-infer", className="me-3", color="danger"),
            ]
        ),
        html.Div(id='parser-data-meta', className="mt-3"),
        dcc.Store(id='parser-infer-state', data=False),
        dcc.Store(id='parser-data', data=train_data[0]),
        dbc.Card(
            [
                dbc.CardHeader("Synthesis Text"),
                dbc.CardBody(id="parser-input"),
            ],
            className="mt-3"
        ),
        dbc.Row(
            [
                html.Div(
                    get_json_viewer_card(
                        viewer_cid="parser-ref-output_reaction_inputs",
                        header="Reference: ReactionInputs",
                    ),
                    className="col-6"
                ),
                html.Div(
                    get_json_viewer_card(
                        viewer_cid="parser-output-output_reaction_inputs",
                        header="LLM: ReactionInputs",
                        data_string="RUN INFERENCE!"
                    ),
                    className="col-6"
                ),
            ],
            className="mt-3"
        ),
        dbc.Row(
            [
                html.Div(
                    get_json_viewer_card(
                        viewer_cid="parser-ref-output_reaction_conditions",
                        header="Reference: ReactionConditions",
                    ),
                    className="col-6"
                ),
                html.Div(
                    get_json_viewer_card(
                        viewer_cid="parser-output-output_reaction_conditions",
                        header="LLM: ReactionConditions",
                        data_string="RUN INFERENCE!"
                    ),
                    className="col-6"
                ),
            ],
            className="mt-3"
        ),
    ]
)


@app.callback(
    Output('parser-data-meta', 'children'),
    Input('parser-data', 'data')
)
def update_meta(data):
    card = dbc.Card(
        [
            dbc.CardHeader('Loaded Data'),
            dbc.CardBody(
                dcc.Markdown(
                    f"""
                  __Reaction ID__:  [{data["reaction_id"]}](https://open-reaction-database.org/client/id/{data["reaction_id"]})
                  
                  __In training?__ {data["reaction_id"] in train_data_ids}
                  """
                ),
            ),
        ]
    )
    return card


@app.callback(
    Output("parser-input", "children"),
    Output("parser-ref-output_reaction_conditions", "data"),
    Output("parser-ref-output_reaction_inputs", "data"),
    Input("parser-data", "data"),
)
def update_input_ref(data):
    return data['input_text'], data['output_reaction_conditions'], data['output_reaction_inputs']


@app.callback(
    Output("parser-data", "data"),
    Output("parser-output-output_reaction_conditions", "data"),
    Output("parser-output-output_reaction_inputs", "data"),
    State("parser-data", "data"),
    Input("parser-load-button-infer", "n_clicks"),
    Input("parser-load-button-test", "n_clicks"),
    Input("parser-load-button-train", "n_clicks"),
    prevent_initial_call=True,
)
def load_and_infer(data, click_infer, click_test, click_train):
    if ctx.triggered_id == "parser-load-button-test" and click_test:
        d = random.choice(test_data)
        return d, "RUN INFERENCE!", "RUN INFERENCE!"
    elif ctx.triggered_id == "parser-load-button-train" and click_train:
        d = random.choice(train_data)
        return d, "RUN INFERENCE!", "RUN INFERENCE!"
    elif ctx.triggered_id == "parser-load-button-infer" and click_infer:
        input_text = data['input_text']
        try:
            completion_text = get_completion(input_text)
        except:
            return data, "", "FAILED TO GET COMPLETION, IS YOUR KEY ALRIGHT?"
        try:
            res = json.loads(completion_text)
            return data, res['output_reaction_conditions'], res['output_reaction_inputs']
        except:
            return data, "", "NOT A VALID JSON!\n\n" + completion_text, True
