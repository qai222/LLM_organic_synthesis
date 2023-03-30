import os

import dash_bootstrap_components as dbc
import flask
from dash import Dash, html, page_registry, page_container, Output, Input, State

os.environ['LOGURU_LEVEL'] = 'WARNING'

BAR_STYLE = {
    "zIndex": "1",
}


def get_navbar(nav_links: list[dbc.NavLink], navbar_id="pvis_navbar"):
    bar = dbc.Row(
        [
            dbc.Col(nl, width="auto") for nl in nav_links
        ],
        className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
        align="right",
    )

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
                dbc.NavbarToggler(id=f"{navbar_id}-toggler", n_clicks=0),
                dbc.Collapse(
                    bar,
                    id=f"{navbar_id}-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ],
        ),
        color="#C0C0C0",
        dark=False,
        className="mb-3",
        style=BAR_STYLE,
    )
    return navbar


def navbar_callback(app, navbar_id: str):
    app.callback(
        Output(f"{navbar_id}-collapse", "is_open"),
        [Input(f"{navbar_id}-toggler", "n_clicks")],
        [State(f"{navbar_id}-collapse", "is_open")],
    )(simple_open)


def simple_open(n, is_open):
    if n:
        return not is_open
    return is_open


def create_dashapp(prefix="/"):
    server = flask.Flask(__name__)

    app = Dash(
        name=__name__,
        title="Synthesis Parser",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        use_pages=True,
        server=server,
        suppress_callback_exceptions=True,
        assets_ignore=r'defer[A-z]*.js',
        url_base_pathname=prefix,
    )

    app_folder = os.path.dirname(os.path.abspath(__file__))
    app._favicon = os.path.join(app_folder, "assets/favicon.ico")

    nav_links = []
    for page in page_registry.values():
        description = page['description']
        href = page['relative_path']
        nav_link = dbc.NavLink(
            description, href=href, className="mx-2 text-dark", active="exact",  # style={"color": "#ffffff"}
        )
        nav_links.append(nav_link)

    navbar = get_navbar(nav_links, "DASH_CID_NAVBAR")
    navbar_callback(app, "DASH_CID_NAVBAR")

    CONTENT_STYLE = {
        "marginLeft": "2rem",
        "marginRight": "2rem",
        "padding": "1rem 1rem",
        "zIndex": "0",
    }

    content = html.Div(id="page-content", children=[page_container], style=CONTENT_STYLE)

    app.layout = html.Div(
        [
            navbar,
            content,
        ],
        # trick from https://stackoverflow.com/questions/35513264
        style={"width": "100vw"}
    )
    return app

APP = create_dashapp()
server = APP.server

if __name__ == '__main__':
    APP.run(host="0.0.0.0", port=8080, debug=True)
