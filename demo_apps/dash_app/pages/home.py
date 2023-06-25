from dash import html, get_app, register_page

register_page(__name__, path='/', description="Home")

layout = html.Div(
    [
        html.H3([
            "Project description",
        ]),
        html.Hr(),
    ]
)

app = get_app()
